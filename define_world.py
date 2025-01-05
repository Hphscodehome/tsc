from topology.from_config_to_topoplogy import net_2_struct
from datatype.define_intersection_class import Intersection
class World():
    def __init__(self, net):
        lane_2_shape, intersection_2_updownstream, lane_2_updownstream = net_2_struct(net)
        self.inters = [Intersection(intersection_id,intersection_2_updownstream,lane_2_shape,lane_2_updownstream) for intersection_id in intersection_2_updownstream.keys()]
        for item in self.inters:
            print(item.traffic_light)
if __name__ == '__main__':
    net = '/data/hupenghui/tsc/net.net.xml'
    World(net)