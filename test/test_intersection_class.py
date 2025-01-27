from datatype.define_intersection_class import Intersection
from topology.from_config_to_topoplogy import net_2_struct
import libsumo as traci
import sumolib
import time

# 读取文件
# 然后将其中的交叉口设置为交叉口对象
if __name__=='__main__':
    start_time = time.time()
    path = '/data/hupenghui/Self/tsc/data/syn1_1x1_1h/data.net.xml'
    net_config = net_2_struct(path)
    intersections = list(net_config.intersection_2_updownstream.keys())
    print(net_config,intersections)
    intersection_id = intersections[0]
    inter = Intersection(traci,intersection_id,\
        position = net_config.intersection_2_position[intersection_id],\
        intersection_2_updownstream = net_config.intersection_2_updownstream,\
        lane_2_shape = net_config.lane_2_shape,\
        lane_2_updownstream = net_config.lane_2_updownstream)
    traci.start([sumolib.checkBinary('sumo'), '-c', '/data/hupenghui/Self/tsc/data/syn1_1x1_1h/data.sumocfg'])
    for i in range(100):
        traci.simulationStep()
        print(inter.get_observation())
        inter.renew()
        print(inter.vehicles)
        if i % 20 == 0:
            print(inter.get_all_info())
    traci.close()
    print(f"used time:{time.time()-start_time}")
    