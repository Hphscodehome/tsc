#region other-package
import traci
import xml.etree.ElementTree as ET
import os
import sumolib
#endregion

#region self-package
from topology.from_config_to_topoplogy import net_2_struct
from datatype.define_intersection_class import Intersection
#endregion

class World():
    def __init__(self, sumocfg):
        self.sumocfg = sumocfg
        tree = ET.parse(self.sumocfg)
        root = tree.getroot()
        self.net_file = os.path.join(os.path.dirname(self.sumocfg),root.find('./input/net-file').get('value'))
        self.route_file = os.path.join(os.path.dirname(self.sumocfg),root.find('./input/route-files').get('value'))
        netconfig = net_2_struct(self.net_file)
        lane_2_shape=netconfig.lane_2_shape
        intersection_2_updownstream=netconfig.intersection_2_updownstream
        lane_2_updownstream=netconfig.lane_2_updownstream
        intersection_2_position=netconfig.intersection_2_position
        self.eng = traci
        self.inters = [Intersection(self.eng,intersection_id,intersection_2_position[intersection_id],intersection_2_updownstream,lane_2_shape,lane_2_updownstream) for intersection_id in intersection_2_updownstream.keys()]
        self.cmd = [sumolib.checkBinary('sumo'), '-c', self.sumocfg, "--remote-port", "8813"]
    
if __name__ == '__main__':
    sumocfg = '/data/hupenghui/tsc/net.sumocfg'
    world = World(sumocfg)
    print(__file__)
    print(__name__)
    print(world.inters[0].upstream_lanes)