#region other-package
import asyncio
from functools import partial
from collections import defaultdict
import numpy as np
import logging
import torch
import math
#endregion

#region my-package
from datatype.define_datatype import Phase,Indicators,Vehicle
from utils.position import judge_cross,calculate_distance
from utils.str_int import get_int,get_char
from utils.constants import Chars
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
        self.vehicles = defaultdict(lambda : Vehicle())
        self.last_step_vehicles = []
        self.all_obs_fn = {
            "lane_average_speed": self.get_lane_average_speed,
            "lane_vehicle_numbers": self.get_lane_vehicle_numbers,
            "lane_halting_numbers": self.get_lane_halting_numbers,
            "lane_waiting_time": self.get_lane_waiting_time,
            "vehicle_map": self.get_vehicle_map, 
            "current_phase": self.get_current_phase
        }
        self.obs_fn = ['vehicle_map','current_phase','lane_waiting_time','lane_halting_numbers','lane_vehicle_numbers','lane_average_speed']
    
    #region 设置相位 
    def get_phase(self,action):
        # action lanes*2
        # lanes*1是link重要性
        # lanes*2是link变不变
        # 结合冲突车道确定下一个可以选择的link。
        phase = self.get_current_phase()
        phase_str = phase.phase_str
        result = ['' for _ in range(len(phase_str))]
        mask = torch.tensor([False for _ in range(len(phase_str))])
        #logging.info(action)
        logits = action[:,0].clone().detach() #torch.tensor(action[:,0])
        #logging.info(logits)
        while '' in result:
            filtered_logits = logits[~mask]  # 取反mask，保留False对应的logits
            indices = torch.arange(len(mask))[~mask]  # 获取mask为False的索引
            distribution = torch.distributions.Categorical(logits=filtered_logits)
            lane_sample = distribution.sample().item()
            lane_sample = indices[lane_sample].item()
            _id = get_int(phase_str[lane_sample])
            probability = torch.sigmoid(action[lane_sample,1])
            distribution = torch.distributions.Bernoulli(probability)# 创建一个Bernoulli分布
            change_sample = distribution.sample().item()
            if change_sample == 1:
                _id += 1
                _id = _id % Chars
            lane_char = get_char(_id)
            result[lane_sample] = lane_char
            if lane_char != 'r':
                conflict_lanes =(torch.tensor(self.lanes_conflict_map[lane_sample,:])>0)
                mask = mask | conflict_lanes
                for index,flag in enumerate(conflict_lanes):
                    if flag:
                        result[index] = 'r'
            mask[lane_sample] = True
        self.set_phase = ''.join(result)
        self.eng.trafficlight.setRedYellowGreenState(self.id,self.set_phase)
        
    def step(self,action):
        # action
        self.get_phase(action)
    #endregion
    
    
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
              phase_str = self.eng.trafficlight.getRedYellowGreenState(self.id).lower(),
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
                vehicle_array[idx,0] = 1.0
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
                vehicle_array[idx,0] = 1.0
                vehicle_array[idx,1] = -speed
            vehicle_attr_value[lane] = vehicle_array
        return vehicle_attr_value
    #endregion
    
    
    #region observe
    def get_observation(self):
        #link 级别的观测结果
        func_state = defaultdict(list)
        func_state_final = defaultdict(lambda: np.array([]))
        for f in self.obs_fn:
            obs = self.all_obs_fn[f]()
            for index,link in enumerate(self.traffic_light_lanes):
                up_lane = self.lane_2_updownstream[link]['from']
                down_lane = self.lane_2_updownstream[link]['to']
                if 'lane' in f:
                    func_state[f].append([obs[up_lane],obs[down_lane]])
                elif 'vehicle' in f:
                    func_state[f].append(np.hstack((obs[up_lane],obs[down_lane])))
                else:
                    func_state[f].append(get_int(obs.phase_str[index]))
            func_state_final[f] = np.stack(func_state[f],axis=0)
        func_state_final['mask'] = self.lanes_conflict_map
        return func_state_final
    #endregion
    
    
    #region 更新
    def renew(self):
        vehicles = []
        for lane in self.upstream_lanes+self.downstream_lanes:
            vehicles.extend(list(self.eng.lane.getLastStepVehicleIDs(lane)))
        for veh in vehicles:
            if self.eng.vehicle.getWaitingTime(veh) != 0:
                self.vehicles[veh].AccumulatedWaitingTime += 1
    #endregion
    
    
    #region done
    def get_done(self):
        return False
    #endregion
    
    
    #region 奖励
    def get_all_info(self):
        # 上个时间段离开路网的车辆数量 辆数
        # 上个时间段离开路网的车辆通过路网的平均停车等待时间 秒每辆
        # 上个时间段内离开路网的车辆数量与路网内等待的车辆数量
        # 上个时间段内等待时间的增加程度
        vehicles = []
        for lane in self.upstream_lanes+self.downstream_lanes:
            vehicles.extend(list(self.eng.lane.getLastStepVehicleIDs(lane)))
        laststep_vehicles = self.last_step_vehicles
        total_vehicles = list(set(vehicles) | set(self.last_step_vehicles))
        
        getin_vehicles = list(set(vehicles) - set(self.last_step_vehicles))
        
        leaved_vehicles = list(set(self.last_step_vehicles) - set(vehicles))
        # 总延迟，并非当前动作的结果，是累积动作的结果
        total_delay = 0
        for veh in leaved_vehicles:
            total_delay += self.vehicles[veh].AccumulatedWaitingTime
        if len(leaved_vehicles) != 0:
            average_delay = total_delay/len(leaved_vehicles)
        else:
            average_delay = 0
        # 吞吐量，是当前动作的结果
        throughput = len(leaved_vehicles)
        # 车辆等待时间的增长情况是当前动作的结果
        laststep_total = 0
        for veh in laststep_vehicles:
            laststep_total += self.vehicles[veh].AccumulatedWaitingTime
        thisstep_total = 0
        for veh in total_vehicles:
            thisstep_total += self.vehicles[veh].AccumulatedWaitingTime
        wait_time_ascend = thisstep_total - laststep_total
        
        self.last_step_vehicles = vehicles
        
        return Indicators(total_vehicles = len(total_vehicles),
                          wait_time_ascend = wait_time_ascend,
                          throughput = throughput,
                          average_delay = average_delay)
    #endregion
    
    
    #region reward
    def get_reward(self):
        indicator = self.get_all_info()
        reward = 0
        if indicator.wait_time_ascend > 0:
            if indicator.throughput == 0:
                reward = -math.log(indicator.wait_time_ascend,7)
            else:
                # 通行比例大则奖励大
                reward = - math.log(indicator.wait_time_ascend,7) * (1 - (indicator.throughput / indicator.total_vehicles))
        else:
            reward = 2
        #reward += -0.7*indicator.average_delay
        #reward += 0.3*indicator.throughput
        return reward, indicator
    #endregion
    
    
if __name__ == '__main__':
    True
