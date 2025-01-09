#region other-package
import gymnasium as gym
import traci
import xml.etree.ElementTree as ET
import os
import sumolib
from collections import defaultdict
#endregion

#region my-package
from topology.from_config_to_topoplogy import net_2_struct
from datatype.define_intersection_class import Intersection
from datatype.define_datatype import Vehicle,Indicators
#endregion

class World(gym.Env):
    def __init__(self, sumocfg):
        self.sumocfg = sumocfg
        tree = ET.parse(self.sumocfg)
        root = tree.getroot()
        self.net_file = os.path.join(os.path.dirname(self.sumocfg),root.find('./input/net-file').get('value'))
        self.route_file = os.path.join(os.path.dirname(self.sumocfg),root.find('./input/route-files').get('value'))
        netconfig = net_2_struct(self.net_file)
        
        self.lane_2_shape=netconfig.lane_2_shape
        self.intersection_2_updownstream=netconfig.intersection_2_updownstream
        self.lane_2_updownstream=netconfig.lane_2_updownstream
        self.intersection_2_position=netconfig.intersection_2_position
        dict_net = netconfig.model_dump()
        for keyr in dict_net.keys():
            print(keyr,dict_net[keyr])
            
        self.eng = traci
        self.cmd = [sumolib.checkBinary('sumo'), '-c', self.sumocfg]
        
        self.inters = [Intersection(self.eng, intersection_id, self.intersection_2_position[intersection_id], self.intersection_2_updownstream, self.lane_2_shape, self.lane_2_updownstream) for intersection_id in self.intersection_2_updownstream.keys()]
        self.vehicles = defaultdict(Vehicle)
        self.last_step_vehicles = []
        self.action_interval = 10
        self.reset()
    
    def close(self):
        self.eng.close()
    
    def reset(self):
        try:
            self.close()
        except:
            print('还没开始')
        self.eng.start(self.cmd)
        self.inters = [Intersection(self.eng, intersection_id, self.intersection_2_position[intersection_id], self.intersection_2_updownstream, self.lane_2_shape, self.lane_2_updownstream) for intersection_id in self.intersection_2_updownstream.keys()]
        self.vehicles = defaultdict(Vehicle)
        self.last_step_vehicles = []
        
        self.n_agent = len(self.inters)
        state = self._get_observations()
        
        return state
    
    def step(self, action):
        # action: Dict[str,lanes*12]
        for item in self.inters:
            item.step(action[item.id])
        for _ in range(self.action_interval):
            self.eng.simulationStep()
            self.renew()
        obs = self._get_observations()
        dones = self._get_dones()
        rewards, infos = self._get_reward_info()
        return obs, rewards, dones, infos
        
    def renew(self):
        vehicles = self.eng.vehicle.getIDList()
        for veh in vehicles:
            if self.eng.vehicle.getWaitingTime(veh) != 0:
                self.vehicles[veh].AccumulatedWaitingTime += 1
        for index, inter in enumerate(self.inters):
            inter.renew()
            
    def _get_observations(self):
        observations = defaultdict(dict)
        for index, inter in enumerate(self.inters):
            observations[inter.id] = inter.get_observation()
        return observations
    
    def _get_reward_info(self):
        rewards = defaultdict(float)
        infos = defaultdict(Indicators)
        for index, inter in enumerate(self.inters):
            reward, indicator = inter.get_reward()
            rewards[inter.id] = reward
            infos[inter.id] = indicator
        indicator = self.get_all_info()
        infos['global'] = indicator
        return rewards, infos
    
    def _get_dones(self):
        dones = defaultdict(bool)
        for index, inter in enumerate(self.inters):
            dones[inter.id] = inter.get_done()
        return dones
    
    #region global_reward
    def get_all_info(self):
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
            average_delay = 0.0
        throughput = len(leaved_vehicles)
        self.last_step_vehicles = vehicles
        return Indicators(throughput = throughput, average_delay = average_delay)
    #endregion
    
    
if __name__ == '__main__':
    sumocfg = '/data/hupenghui/TSC/data/syn1_1x1_1h/data.sumocfg'
    world = World(sumocfg)
    print(__file__)
    print(__name__)
    print(world.inters[0].id)
    print(world.inters[0].upstream_lanes)
    print(world.inters[0].downstream_lanes)
    print(world.inters[0].lanes_conflict_map)
    world.close()
    