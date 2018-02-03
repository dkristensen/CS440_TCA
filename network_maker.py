from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import subprocess
import random
import math

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

NETWORK_EXTENSION = ".net.xml"
ROUTES_EXTENSION = ".rou.xml"


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--height", dest="height", type=int,
                         default=5, help="How many traffic lights there should be going NS")
    optParser.add_option("--width", dest="width", type=int,
                         default=5, help="How many traffic lights there should be going EW")
    optParser.add_option("--hdist", dest="hlength", type=int,
                         default=100, help="The length of each street segment between two adjacent lights going NS")
    optParser.add_option("--vdist", dest="vlength", type=int,
                         default=100, help="The length of each street segment between two adjacent lights going EW")
    optParser.add_option("--fn", dest="filename", type=str,
                         default="default", help="The filename to save the network into")
    optParser.add_option("--start", dest="startTime", type=int,
                         default=0, help="The time to start the simulation from (s)")
    optParser.add_option("--end", dest="endTime", type=int,
                         default=3600, help="The time to end the simulation at (s)")
    optParser.add_option("--max_arrivals", dest="maxArrivals", type=int,
                         default=2, help="The maximum number of cars to bring in at each time step")
    optParser.add_option("--vehicle_file", dest="vehicleFile", type=str,
                         default="vehicles.add.xml", help="File holding declerations of different vehicles")
    optParser.add_option("--tls", dest="tls", action="store_false",
                        default=True, help="Create the network without traffic lights")
    optParser.add_option("--grid", dest="isGrid", action="store_false",
                        default=True, help="Create the network with a grid layout")
    options, args = optParser.parse_args()
    return options

def get_angle(currJunc, otherJunc):
    '''
    Gets the angle between the current Junction and another junction, relative to the x-axis
    '''
    # Get the angle in radians and mulitply by 2 since not giving the correct value
    angle = math.atan2(otherJunc[1]-currJunc[1], otherJunc[0]-currJunc[0])*2
    return math.degrees(angle)

def parse_junction_line(line, height, width):
    # Split the junction line up to get the id
    jid = line.split("<junction id=\"")[1].split("\"")[0]
    # Try to split the id at the / in the case of the central nodes from a grid layout
    split_id = jid.split('/')
    x_val = -1
    y_val = -1
    if(len(split_id)>1):
        x_val = int(split_id[0])+1
        y_val = int(split_id[1])+1
    else:
        if('bottom' in jid):
            y_val = 0
            x_val = int(jid.split('bottom')[1])+1
        elif('top' in jid):
            y_val = height+1
            x_val = int(jid.split('top')[1])+1
        elif('left' in jid):
            y_val = int(jid.split('left')[1])+1
            x_val = 0
        elif('right' in jid):
            y_val = int(jid.split('right')[1])+1
            x_val = width+1
    return (jid,x_val,y_val)
    
def format_network(filename, height, width):
    # read in the file that we just generated to edit the names to match the naming
    # conventions for the project
    with open(filename) as f:
        contents = f.readlines()

    jcount = 0
    junction_names = []
    old_ids = []

    for line in contents:
        if( "<junction id=" in line and "type=\"internal\"" not in line):
            old_ids.append(parse_junction_line(line, height, width))

    # We need to sort from the highest junction id's downwards, so that we dont overwrite partial names (like the 1/3 in 11/3)     
    old_ids = reversed(sorted(old_ids, key=lambda x: (x[1],x[2])))

    for old_id,_,_ in old_ids:
        # For each line declaring a junction, reformat the junction id to be the number that
        # we encountered it in this search. Replace every instance of it in the contents
        # with the new junction id
        new_junc_id = "J{}".format(('{:04d}'.format(jcount)))
        # print(jid, new_junc_id)
        contents = [line.replace(old_id,new_junc_id) for line in contents]
        junction_names.append(new_junc_id)
        jcount+=1
    
    for junction in junction_names:
        # Get the list of every edge with tag "to" set to the current junction
        incoming_edges = [line for line in contents if "to=\"{}\"".format(junction) in line and "<edge" in line]
        incoming_nodes = []

        # For each edge incoming to our junction, get the junction the edge is coming from
        for edge in incoming_edges:
            incoming_nodes.append(edge.split("from=\"")[1].split("\"")[0])
        
        inc_loc = []
        # For each junction attached to the current junction, get the (x,y) locations
        for node in incoming_nodes:
            junc = [line for line in contents if "<junction id=\"{}".format(node) in line][0]
            x = float(junc.split("x=\"")[1].split("\"")[0])
            y = float(junc.split("y=\"")[1].split("\"")[0])
            inc_loc.append((x,y))
        
        # Get current junction and its (x,y) location
        my_junc = [line for line in contents if "<junction id=\"{}\"".format(junction) in line][0]
        my_x = float(my_junc.split("x=\"")[1].split("\"")[0])
        my_y = float(my_junc.split("y=\"")[1].split("\"")[0])
        
        # For each junction connected to the current junction, calculate the angle between
        # the current junction and the connected junction
        degrees = []
        for loc in inc_loc:
            degrees.append(get_angle((my_x,my_y), loc))

        # Sort the junctions by their radians around the current junction
        ordered_edges = [edge for _,edge in sorted(zip(degrees,incoming_edges))]

        # rename the edge to be <currentJunctionid>e_<sortedIndex>
        for i in range(len(ordered_edges)):
            substring = ordered_edges[i].split("id=\"")[1].split("\"")[0]
            contents = [line.replace(substring,"{}e_{}".format(junction,i)) for line in contents]

    # Rewrite all of our edits back to the file to overwrite
    with open(filename,'w') as f:
        for line in contents:
            f.write(line)

    return

def generate_grid_network(options):
    DEFAULT_CONNECTION_LENGTH = 100
    height = options.height
    width = options.width
    hlength = options.hlength
    vlength = options.vlength
    filename = options.filename+NETWORK_EXTENSION
    trafficLight = options.tls
    os.system('''netgenerate --grid \
                            --grid.x-number={x} \
                            --grid.y-number={y} \
                            --grid.x-length={xl} \
                            --grid.y-length={yl} \
                            --output-file={fn} \
                            --grid.attach-length={al} \
                            --tls.guess={tls}'''.format(
                                    x=width,
                                    y=height,
                                    xl=hlength,
                                    yl=vlength,
                                    fn=filename,
                                    al=DEFAULT_CONNECTION_LENGTH,
                                    tls=trafficLight))
    format_network(filename, height, width)
    

def generate_routefile(options):
    currentPath = "./tools/"
    startTime = options.startTime
    endTime = options.endTime
    maxSimultaneousArrivals = options.maxArrivals
    seed = 42
    fringeFactor = 10.0
    period = 1.0
    minDistance = 400.0
    network=options.filename+NETWORK_EXTENSION
    routeFile = options.filename+ROUTES_EXTENSION
    vehicleFile=options.vehicleFile
    os.system("python2 {path}randomTrips.py \
                        -n {netFile} \
                        -r {routeFile} \
                        --seed {seed} \
                        -b {begin} \
                        -e {end} \
                        -p {period} \
                        --binomial {N}\
                        --validate \
                        --fringe-factor {fringe}".format(
                                path=currentPath,
                                N=maxSimultaneousArrivals,
                                begin=startTime,
                                end=endTime,
                                period=period,
                                seed=seed,
                                fringe=fringeFactor,
                                minDist=minDistance,
                                netFile=network,
                                routeFile=routeFile,
                                addFile= vehicleFile
                                ))


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()
    
    # Generate the network
    if(options.isGrid):
        generate_grid_network(options)
    

    # generate the routes for this simulation
    generate_routefile(options)

    # Create the config file to launch the simulation from
    with open(options.filename+".sumocfg", "w") as config_file:
        print('''<?xml version="1.0" encoding="UTF-8"?>

<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    
    <input>
        <net-file value=\"{}\"/>
        <route-files value=\"{}\"/>
    </input>

    <time>
        <begin value="0"/>
    </time>

    <report>
        <verbose value="true"/>
        <no-step-log value="true"/>
    </report>  

</configuration>'''.format(options.filename+NETWORK_EXTENSION, options.filename+ROUTES_EXTENSION), file=config_file)

    

  