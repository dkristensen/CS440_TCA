"""
File holding all my scripts for getting the state and reward for the network
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import subprocess
import random

# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__),"tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci

# Threshold in m/s for cars satisfying our state space condition
SPEED_THRESHOLD = 5
# Max number of possible light phases
LIGHT_CLASSES = 12

def get_all_traffic_light_ids():
    """Returns a list holding the ids for all the traffic lights in the system"""
    # only serves as a wrapper for the TraCI command
    return traci.trafficlight.getIDList()


def get_controlled_lanes(traffic_light_id):
    """
    Returns a list of the lanes that the traffic light controlls broken up by incoming direction
    :param traffic_light_id: the id for the traffic light in the network
    :return: A list holding the lane ids for each incoming direction
    """
    # Get traffic lanes controlled by the light and get rid of duplicates
    controlled_lanes = traci.trafficlight.getControlledLanes(traffic_light_id)
    controlled_lanes = set(controlled_lanes)
    
    # Set the layout for assigning lanes
    directional_lanes = []
    directional_name = ['e_0','e_1','e_2','e_3','e_4','e_5']
    
    # For each direction, assign the lane that is coming in from 
    # that direction to the corresponding lane array
    for name in directional_name:
        directional_lanes.append([lane for lane in controlled_lanes if name in lane])
    return directional_lanes


def get_wait_time(controlled_lanes):
    """
    Returns the wait time for each direction on the controlled lanes for the traffic light
    Currently linear, since it is a much more elegant implementation. 
    Need to work to make a non-linear wait time function
    :param controlled_lanes: A list holding the lane ids controlled by a traffic light ordered by incoming direction
    :return: A list with 1:1 correspondence of the lane ids input to the wait of all vehicles occupying the lane
    """
    lane_waits = []
    # For each direction, get the waiting time for each lane in the direction and append to lane_waits
    for i in range(len(controlled_lanes)):
        lane_waits.append(map(traci.lane.getWaitingTime, controlled_lanes[i]))
    return lane_waits



def get_wait_time_for_light(traffic_light_id):
    """
    Gets the linear wait time for the lanes controlled by the traffic light
    :param traffic_light_id: The id of the traffic light to get wait times from
    :return: list holding the wait times for the incoming lanes controlled by the tl
    """
    # Get the lanes
    controlled_lanes = get_controlled_lanes(traffic_light_id)
    # Get the waits
    waits = get_wait_time(controlled_lanes)
    return waits

def get_network_waiting_time(summed=False):
    """
    Returns waiting times for all lanes in the network, seperated by tl and incoming edge    
    :param summed: a boolean (false by default) that controlls whether or not to sum all the lanes' waiting times together 
    :return: A list holding the waiting times for each lane controllde by each light in the network
    """
    # Get the ids of all the traffic lights in the network
    network_lights = get_all_traffic_light_ids()
    # Pass the array of ids through get_wait_time_for_light function using map(f,x)
    traffic_light_waits = map(get_wait_time_for_light, network_lights)
    if(summed):
        for light in range(len(traffic_light_waits)):
            traffic_light_waits[light] = map(sum, traffic_light_waits[light])
    return traffic_light_waits


def get_network_state():
    """Returns the state for all lights in the network"""
    # Get the ids of all the traffic lights in the network
    network_lights = get_all_traffic_light_ids()
    network_lanes = map(get_controlled_lanes, network_lights)
    # Get the list of all cars in each lane for the traffic lights
    cars_in_lanes = map(get_num_cars_in_lanes, network_lanes)
    # Get the list of all cars in each lane under a threshold for the traffic lights
    cars_under_thresh = map(get_cars_under_threshold_at_light, network_lanes)
    # Get the list of all current phases for the traffic lights
    current_phases = map(get_traffic_light_phase, network_lights)
    # Zip them all together so we can have each entry containing the state space for that light
    state_space = zip(cars_in_lanes,
                        cars_under_thresh,
                        current_phases)
    return state_space

def get_flattened_states():
    """Returns the phase of the traffic light in the form of a one hot vector"""
    states = get_network_state()
    flattened = [data for state in states for data in state]
    return flattened

def get_num_cars_in_lanes(lanes):
    """
    Returns the number of cars in each lane at the light 
    :param lanes: the lanes of the traffic light to get the cars from
    :return: the number of cars moving less than our threshold in each lane
    """
    num_cars = []
    for direction in lanes:
        cars = map(traci.lane.getLastStepVehicleNumber, direction)
        if(not cars):
            cars = [0]*len(lanes[0])
        num_cars.append(cars)
    return num_cars

def get_cars_under_threshold_at_light(lanes, thresh=SPEED_THRESHOLD):
    """
    Returns the number of cars in each lane moving less than some speed at the light
    :param lanes: the lanes of the traffic light to get the cars from
    :return: the number of cars moving less than our threshold in each lane
    """
    num_under_thresh = []
    for direction in lanes:
        current_direction = []
        for lane in direction:
            # Get the vehicles on each lane and their speeds
            vh_ids = traci.lane.getLastStepVehicleIDs(lane)
            vh_speeds = map(traci.vehicle.getSpeed, vh_ids)
            # Get the number of cars that are going under the threshold declared at the top of the file
            current_direction.append(len([car for car in vh_speeds if car<=thresh]))
        num_under_thresh.append(current_direction)
        if(not num_under_thresh[-1]):
            num_under_thresh[-1] = [0]*len(lanes[0])
    return num_under_thresh

def get_traffic_light_phase(trafficLightID):
    """
    Returns the phase of the traffic light in the form of a one hot vector
    :param trafficLightID: The id of the traffic light to get the phase of
    :return: A one hot list holding the current traffic state
    """
    phase = traci.trafficlight.getPhase(trafficLightID)
    phase_vector = [0]*LIGHT_CLASSES
    phase_vector[phase] = 1
    return phase_vector

