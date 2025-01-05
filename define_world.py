from topology.from_config_to_topoplogy import net_2_struct
from datatype.define_intersection_class import Intersection
import traci
import xml.etree.ElementTree as ET
import os
import sumolib
class World():
    def __init__(self, sumocfg):
        self.sumocfg = sumocfg
        tree = ET.parse(self.sumocfg)
        root = tree.getroot()
        self.net_file = os.path.join(os.path.dirname(self.sumocfg),root.find('./input/net-file').get('value'))
        self.route_file = os.path.join(os.path.dirname(self.sumocfg),root.find('./input/route-files').get('value'))
        lane_2_shape, intersection_2_updownstream, lane_2_updownstream = net_2_struct(self.net_file)
        self.inters = [Intersection(intersection_id,intersection_2_updownstream,lane_2_shape,lane_2_updownstream) for intersection_id in intersection_2_updownstream.keys()]
        self.eng = traci
        self.cmd = [sumolib.checkBinary('sumo'), '-c', self.sumocfg, "--remote-port", "8813"]
    
if __name__ == '__main__':
    net = '/data/hupenghui/tsc/net.sumocfg'
    World(net)
    print(__file__)
    print(__name__)