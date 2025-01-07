#region other-package
import asyncio
from functools import partial
from collections import defaultdict
import numpy as np
#endregion

#region temp
import logging
#endregion

#region my-package
from datatype.define_datatype import Phase
from utils.position import judge_cross,calculate_distance
#endregion

class Intersection():
    def __init__(self,eng,intersection_id,position,intersection_2_updownstream,lane_2_shape,lane_2_updownstream):
        super().__init__()
        self.eng = eng
        self.id = intersection_id
        self.position = position
        self.upstream_lanes = list(intersection_2_updownstream[self.id]['from'])
        self.downstream_lanes = list(intersection_2_updownstream[self.id]['to'])
        self.traffic_light_lanes = [item[1] for item in sorted(list(intersection_2_updownstream[self.id]['traffic_light']), key=lambda x:x[0])]
        self.lane_2_shape = {lane:lane_2_shape[lane] for lane in self.traffic_light_lanes}
        self.lane_2_updownstream = {lane:lane_2_updownstream[lane] for lane in self.traffic_light_lanes}
        self.lanes_conflict_map = self.get_lanes_conflict_map()
        # 全局设置
        self.vehicle_gap = 7.5
        self.max_vehicle_num = 40
        self.veh_list = []
        
        
    #region 统计冲突车道
    def get_lanes_conflict_map(self):
        # 找到各个车道之间哪个与哪个是冲突的。根据坐标信息。
        lanes_conflict_map = []
        for lane_index_i in range(len(self.traffic_light_lanes)):
            lane_conflict_map = []
            for lane_index_j in range(len(self.traffic_light_lanes)):
                if lane_index_i == lane_index_j:
                    lane_conflict_map.append(0)
                else:
                    lane_i = self.traffic_light_lanes[lane_index_i]
                    lane_j = self.traffic_light_lanes[lane_index_j]
                    if judge_cross([self.lane_2_shape[lane_i][0],self.lane_2_shape[lane_i][-1]],
                                [self.lane_2_shape[lane_j][0],self.lane_2_shape[lane_j][-1]]):
                        lane_conflict_map.append(1)
                    else:
                        lane_conflict_map.append(0)
            lanes_conflict_map.append(lane_conflict_map)
        return np.array(lanes_conflict_map)
    #endregion 
    
    
    #region 车道
    def get_lane_average_speed(self):
        # 平均速度平均速度越大，说明越不紧急
        lane_attr_value = defaultdict(float)
        for lane in self.upstream_lanes+self.downstream_lanes:
            lane_attr_value[lane] = self.eng.lane.getLastStepMeanSpeed(lane)
        return lane_attr_value

    def get_lane_vehicle_numbers(self):
        # 车辆数量，并非waiting，车辆数量越多，说明路占用率越高。
        lane_attr_value = defaultdict(float)
        for lane in self.upstream_lanes+self.downstream_lanes:
            lane_attr_value[lane] = self.eng.lane.getLastStepVehicleNumber(lane)
        return lane_attr_value
    
    def get_lane_halting_numbers(self):
        # 停止车辆数量，停车就说明在等红绿灯
        lane_attr_value = defaultdict(float)
        for lane in self.upstream_lanes+self.downstream_lanes:
            lane_attr_value[lane] = self.eng.lane.getLastStepHaltingNumber(lane)
        return lane_attr_value
    
    def get_lane_waiting_time(self):
        # 所有车辆等红灯的等待时间
        lane_attr_value = defaultdict(float)
        for lane in self.upstream_lanes+self.downstream_lanes:
            lane_attr_value[lane] = self.eng.lane.getWaitingTime(lane)
        return lane_attr_value
    #endregion
    
    
    #region 道路
    def get_edge_average_speed(self):
        pass
    
    def get_edge_waiting_time(self):
        pass
    #endregion
    
    
    #region 信号灯
    def get_current_phase(self):
        # 返回相位字符串、持续时间、索引index
        '''
        trafficlight_attr_value = {}
        trafficlight_attr_value['phase_str'] = self.eng.trafficlight.getRedYellowGreenState(self.id)
        trafficlight_attr_value['phase_id'] = self.eng.trafficlight.getPhase(self.id)
        trafficlight_attr_value['phase_duration'] = self.eng.trafficlight.getPhaseDuration(self.id)
        '''
        trafficlight_attr_value = Phase(phase_id = self.eng.trafficlight.getPhase(self.id),
              phase_str = self.eng.trafficlight.getRedYellowGreenState(self.id),
              phase_duration = self.eng.trafficlight.getPhaseDuration(self.id))
        return trafficlight_attr_value
    #endregion
    
    
    #region 车辆
    def get_vehicle_map(self):
        # 根据可控车道统计车辆信息
        vehicle_attr_value = defaultdict(lambda : np.array([]))
        for lane in self.upstream_lanes:
            vehicle_array = np.zeros((self.max_vehicle_num,2))
            lane_vehicles = self.eng.lane.getLastStepVehicleIDs(lane)
            for veh in lane_vehicles:
                posxy = self.eng.vehicle.getPosition(veh)
                speed = self.eng.vehicle.getSpeed(veh)
                distance = calculate_distance(posxy,self.position)
                idx = int(distance // self.vehicle_gap)
                if idx >= self.max_vehicle_num:
                    continue
                vehicle_array[idx,0] = distance
                vehicle_array[idx,1] = speed
            vehicle_attr_value[lane] = vehicle_array
        for lane in self.downstream_lanes:
            vehicle_array = np.zeros((self.max_vehicle_num,2))
            lane_vehicles = self.eng.lane.getLastStepVehicleIDs(lane)
            for veh in lane_vehicles:
                posxy = self.eng.vehicle.getPosition(veh)
                speed = self.eng.vehicle.getSpeed(veh)
                distance = calculate_distance(posxy,self.position)
                idx = int(distance // self.vehicle_gap)
                if idx >= self.max_vehicle_num:
                    continue
                vehicle_array[idx,0] = distance
                vehicle_array[idx,1] = -speed
            vehicle_attr_value[lane] = vehicle_array
        return vehicle_attr_value
    #endregion
    
    
    #region 奖励
    def get_throughput_reward(self):
        # 平均吞吐量
        veh_list = []
        for lane in self.upstream_lanes+self.downstream_lanes:
            veh_list.extend(list(self.eng.lane.getLastStepVehicleIDs(lane)))
        cnt = 0
        for veh in veh_list:
            if veh in self.veh_list:
                cnt += 1
        throughput = (len(self.veh_list) - cnt)
        self.veh_list = veh_list
        return throughput
    
    def get_queue_length_reward(self):
        # 不太可靠的指标
        return - np.mean([self.eng.lane.getLastStepHaltingNumber(lane) for lane in self.upstream_lanes+self.downstream_lanes])
    
    def get_delay_reward(self):
        # 上个时间段内的等待时间
        total_waiting_time = 0
        for i, lane in enumerate(self.upstream_lanes+self.downstream_lanes):
            lane_vehicles = self.eng.lane.getLastStepVehicleIDs(lane)
            for veh in lane_vehicles:
                waiting_time = self.eng.vehicle.getWaitingTime(veh)
                total_waiting_time += waiting_time
        return -total_waiting_time
    #endregion
    
    #region test
    def get_observation(self):
        obs = []
        lane_obs = []
        for f in self.obs_fn:
            if f in self.lane_obs_fn:
                lane_obs.append(self.all_obs_fn[f]())
        if len(lane_obs) > 1: 
            obs.append(np.stack(lane_obs, axis=-1))
        
        for f in self.obs_fn:
            if f not in self.lane_obs_fn:
                obs.append(self.all_obs_fn[f]())
        return np.array(obs, dtype='object')
    
    
    def get_reward(self):
        reward = 0
        for f, w in zip(self.reward_fn, self.reward_weight):
            reward += w * self.all_reward_fn[f]()
        return reward
    #endregion
    
if __name__ == '__main__':
    True
