#region other-package
import traci
import xml.etree.ElementTree as ET
import os
import sumolib
from collections import defaultdict
#endregion

#region self-package
from topology.from_config_to_topoplogy import net_2_struct
from datatype.define_intersection_class import Intersection
from datatype.define_datatype import Vehicle,Global_indicators
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
        dict_net = netconfig.dict()
        for keyr in dict_net.keys():
            print(keyr,dict_net[keyr])
        self.eng = traci
        self.inters = [Intersection(self.eng,intersection_id,intersection_2_position[intersection_id],intersection_2_updownstream,lane_2_shape,lane_2_updownstream) for intersection_id in intersection_2_updownstream.keys()]
        self.cmd = [sumolib.checkBinary('sumo'), '-c', self.sumocfg, "--remote-port", "8813"]
        self.vehicles = defaultdict(Vehicle)
        self.last_step_vehicles = []
        
    def step(self):
        self.renew_global()
        pass
    
    def renew_global(self):
        vehicles = self.eng.vehicle.getIDList()
        for veh in vehicles:
            self.vehicles[veh].AccumulatedWaitingTime = self.eng.vehicle.getAccumulatedWaitingTime(veh)
    
    #region global_reward
    def get_throughput_reward(self):
        # 上个时间段离开路网的车辆数量 辆数
        # 上个时间段离开路网的车辆通过路网的平均等待时间 秒每辆
        vehicles = self.eng.vehicle.getIDList()
        leaved_vehicles = list(set(self.last_step_vehicles)-set(vehicles))
        total_delay = 0
        for veh in leaved_vehicles:
            total_delay += self.vehicles[veh].AccumulatedWaitingTime
        if len(leaved_vehicles) != 0:
            average_delay = total_delay/len(leaved_vehicles)
        else:
            average_delay = 0
        throughput = len(leaved_vehicles)
        self.last_step_vehicles = vehicles
        return Global_indicators(throughput = throughput, average_delay = average_delay)
    #endregion
    
if __name__ == '__main__':
    sumocfg = '/data/hupenghui/LibSignal/data/raw_data/hangzhou_4x4_hetero/hangzhou_4x4_gudang_18041610_1h_m.sumocfg'
    world = World(sumocfg)
    print(__file__)
    print(__name__)
    print(world.inters[0].id)
    print(world.inters[0].upstream_lanes)
    print(world.inters[0].downstream_lanes)
    print(world.inters[0].lanes_conflict_map)
    