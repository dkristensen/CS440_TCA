from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import subprocess
import random
import state_getter as sg

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

time_step = 1
time_modifier = 1/time_step

def gen_routes_file():
    random.seed(42)  # make tests reproducible
    N = int(3600 * time_modifier)  # second to sim
    # demand per second from different directions
    pW = 1. / 10 * time_modifier
    pE = 1. / 15 * time_modifier
    pN = 1. / 10 * time_modifier
    pS = 1. / 14 * time_modifier

    vehicle_possibilities = ["CarA", "CarB", "CarC", "CarD"]

    directional_probabilities = [pN, pS, pE, pW]
    directions = ["north", "south", "east", "west"]
    dirColor = ["blue","red","green","yellow"]

    north_routes = ["NS","NE","NW"]
    south_routes = ["SN","SE","SW"]
    east_routes = ["EN","ES","EW"]
    west_routes = ["WN","WS","WE"]
    
    route_order = [north_routes,
                south_routes, 
                east_routes,
                west_routes]

    with open("my_test.rou.xml", "w") as routes:
        print("""<routes>
        <vType accel="3.0" decel="6.0" id="CarA" length="5.0" minGap="2.5" maxSpeed="50.0" sigma="0.5" />
        <vType accel="2.5" decel="6.0" id="CarB" length="7.5" minGap="2.5" maxSpeed="50.0" sigma="0.5" />
        <vType accel="2.0" decel="5.0" id="CarC" length="5.0" minGap="2.5" maxSpeed="40.0" sigma="0.5" />
        <vType accel="1.5" decel="5.0" id="CarD" length="7.5" minGap="2.5" maxSpeed="30.0" sigma="0.5" />
        
        <route id="EW" edges="Ei Wo" />
        <route id="ES" edges="Ei So" />
        <route id="EN" edges="Ei No" />
        
        <route id="WN" edges="Wi No" />
        <route id="WS" edges="Wi So" />
        <route id="WE" edges="Wi Eo" />
        
        <route id="SN" edges="Si No" />
        <route id="SE" edges="Si Eo" />
        <route id="SW" edges="Si Wo" />
        
        <route id="NS" edges="Ni So" />
        <route id="NE" edges="Ni Eo" />
        <route id="NW" edges="Ni Wo" />""", file=routes)
        lastVeh = 0
        vehNr = 0
        for i in range(N):
            for j in range(4):
                if random.uniform(0, 1) < directional_probabilities[j]:
                    route_choice = random.choice(route_order[j])
                    vehicle_choice = random.choice(vehicle_possibilities)
                    print('    <vehicle id="{origin}_{id}" type="{vhType}" route="{route}" depart="{depTime}" color="{color}"/>'.format(
                        origin = directions[j],
                        id=vehNr,
                        vhType=vehicle_choice,
                        route = route_choice,
                        depTime = i,
                        color=dirColor[j]),  file=routes)
                    vehNr += 1
                    lastVeh = i
        print("</routes>", file=routes)
# gen_routes_file()



def run():
    """execute the TraCI control loop"""
    step = 0
    start_change = 0
    duration = 0
    # we start with phase 0 where NS has left green
    traci.trafficlight.setPhase("Origin", 0)
    controlled_lanes = traci.trafficlight.getControlledLanes("Origin")
    # print(sg.get_wait_time_for_light("Origin"))
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        wait_times = sg.get_network_waiting_time(summed=True)[0]
        NS = wait_times[0]+wait_times[2]
        EW = wait_times[1]+wait_times[3]
        if(EW>(20+NS)):
            if(traci.trafficlight.getPhase("Origin") < 2):
                traci.trafficlight.setPhase("Origin", 2)
                start_change = step
                duration=0
            else:
                if(step-start_change>6):
                    if(duration<8):
                        #Left turns for EW
                        traci.trafficlight.setPhase("Origin", 3)
                        duration+=1
                    else:
                        #Straight for EW
                        traci.trafficlight.setPhase("Origin", 4)
        elif(NS>(20+EW)):
            if(traci.trafficlight.getPhase("Origin") == 3 or traci.trafficlight.getPhase("Origin") == 4):
                traci.trafficlight.setPhase("Origin", 5)
                start_change = step
                duration=0
            else:
                if(step-start_change>6):
                    if(duration<8):
                        #Left turns for EW
                        traci.trafficlight.setPhase("Origin", 0)
                        duration+=1
                    else:
                        #Straight for EW
                        traci.trafficlight.setPhase("Origin", 1)
        step += 1

    traci.close()
    sys.stdout.flush()


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # first, generate the route file for this simulation
    gen_routes_file()

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "my_test.sumocfg",
                             "--tripinfo-output", "tripinfo.xml",
                             "--step-length", str(time_step)])
    run()
