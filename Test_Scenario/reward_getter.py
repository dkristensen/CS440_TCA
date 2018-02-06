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



def get_all_traffic_light_ids():
    """
    Returns a list holding the ids for all the traffic lights in the system
    """
    # only serves as a wrapper for the TraCI command
    return traci.trafficlight.getIDList()


def get_controlled_lanes(traffic_light_id):
    """
    Returns a list of the lanes that the traffic light controlls broken up by incoming direction
    of the form [[FromNorth],[FromEast],[FromSouth],[FromWest]]
    ---
    Args:
        traffic_light_id: the id for the traffic light in the network
    ---
    Returns:
        A list holding the lane ids for each incoming direction
    """
    # Get traffic lanes controlled by the light and get rid of duplicates
    controlled_lanes = traci.trafficlight.getControlledLanes(traffic_light_id)
    controlled_lanes = set(controlled_lanes)
    
    # Set the layout for assigning lanes to the typical 4 way intersection
    directional_lanes = []
    directional_names=["N","E","S","W"]
    
    # For each direction, assign the lane that is coming in from 
    # that direction to the corresponding lane array
    for i in range(len(directional_names)):
        directional_lanes.append([lane for lane in controlled_lanes if directional_names[i] in lane])
    return directional_lanes


def get_wait_time(controlled_lanes):
    """
    Returns the wait time for each direction on the controlled lanes for the traffic light
    Currently linear, since it is a much more elegant implementation. 
    Need to work to make a non-linear wait time function
    ---
    Args:
        controlled_lanes: A list holding the lane ids controlled by a traffic light ordered by incoming direction
    ---
    Returns:
        A list with 1:1 correspondence of the lane ids input to the wait of all vehicles occupying the lane
    """
    lane_waits = []
    # For each direction, get the waiting time for each lane in the direction and append to lane_waits
    for i in range(len(controlled_lanes)):
        lane_waits.append(map(traci.lane.getWaitingTime, controlled_lanes[i]))
    return lane_waits



def get_wait_time_for_light(traffic_light_id):
    # Get the lanes
    controlled_lanes = get_controlled_lanes(traffic_light_id)
    # Get the waits
    waits = get_wait_time(controlled_lanes)
    return waits

def get_network_waiting_time(summed=False):
    """
    Returns a list with each traffic light in the network having a sublist holding the wait times for the 
    lanes that the light controls, seperated by incoming direction
    ---
    Args: 
        summed: a boolean (false by default) that controlls whether or not to sum all the lanes' waiting
        times together 
    ---
    Returns:
        A list holding the waiting times for each lane controllde by each light in the network
    """
    # Get the ids of all the traffic lights in the network
    network_lights = get_all_traffic_light_ids()
    # Pass the array of ids through get_wait_time_for_light function using map(f,x)
    traffic_light_waits = map(get_wait_time_for_light, network_lights)
    if(summed):
        for light in range(len(traffic_light_waits)):
            traffic_light_waits[light] = map(sum, traffic_light_waits[light])
    return traffic_light_waits

