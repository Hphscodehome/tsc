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
        self.all_obs_fn = {
            "lane_average_speed": self.get_lane_average_speed,
            "lane_vehicle_numbers": self.get_lane_vehicle_numbers,
            "lane_halting_numbers": self.get_lane_halting_numbers,
            "lane_waiting_time": self.get_lane_waiting_time,
            "vehicle_map": self.get_vehicle_map, 
            "current_phase": self.get_current_phase
        }
        self.obs_fn = ['vehicle_map','current_phase','lane_waiting_time','lane_halting_numbers','lane_vehicle_numbers','lane_average_speed']
        self.vehicles = defaultdict(lambda : Vehicle())
        self.last_step_vehicles = []
        self.last_step_waittime = 0.0
        self.last_step_watinums = 0
        self.leaved_vehicles = 0
        self.leaved_delay = 0
        self.cahnge_phase = False
        
    def set_reset_phase(self):
        phase = self.get_current_phase()
        phase_str = phase.phase_str
        self.eng.trafficlight.setRedYellowGreenState(self.id,'r'*len(phase_str))
    
    #region 设置相位 
    def get_phase(self,action):
        # action lanes*2
        phase = self.get_current_phase()
        phase_str = phase.phase_str
        result = ['' for _ in range(len(phase_str))]
        mask = torch.tensor([False for _ in range(len(phase_str))])
        for i in range(len(phase_str)):
            if phase_str[i] == 'y':
                result[i] = 'r'
                mask[i] = True
        logits = action[:,0].clone().detach() #torch.tensor(action[:,0])
        while '' in result:
            filtered_logits = logits[~mask]  # 取反mask，保留False对应的logits
            indices = torch.arange(len(mask))[~mask]  # 获取mask为False的索引
            lane_sample = torch.argmax(filtered_logits).item()
            lane_sample = indices[lane_sample].item()
            _id = get_int(phase_str[lane_sample])
            probability = torch.sigmoid(action[lane_sample,1])
            change_sample = 1 if probability > 0.5 else 0  # 确定性选择
            if change_sample == 1:
                _id += 1
                _id = _id % Chars
            lane_char = get_char(_id)
            if lane_char != 'r':
                conflict_lanes =(torch.tensor(self.lanes_conflict_map[lane_sample,:])>0)
                can_change = True
                for index,flag in enumerate(conflict_lanes):
                    if flag:
                        if phase_str[index] == 'g':
                            can_change = False
                            break
                if can_change:
                    result[lane_sample] = lane_char
                    mask = mask | conflict_lanes
                    for index,flag in enumerate(conflict_lanes):
                        if flag:
                            result[index] = 'r'
                else:
                    result[lane_sample] = phase_str[lane_sample]
            else:
                result[lane_sample] = lane_char
            mask[lane_sample] = True
        self.set_phase = ''.join(result)
        logging.info(f"next:{''.join(result)}")
        if ''.join(result) != phase_str:
            self.cahnge_phase = True
        else:
            self.cahnge_phase = False
        self.eng.trafficlight.setRedYellowGreenState(self.id,self.set_phase)
        
    def step(self,action):
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
        # 在这个车道上的车辆的数量，不管动或者静止。
        lane_attr_value = defaultdict(float)
        for lane in self.upstream_lanes+self.downstream_lanes:
            lane_attr_value[lane] = self.eng.lane.getLastStepVehicleNumber(lane)
            #logging.info(f"compare vehicle nums:{lane_attr_value[lane]},{len(self.eng.lane.getLastStepVehicleIDs(lane))}")
        return lane_attr_value
    
    def get_lane_halting_numbers(self):
        # 车道的停车数量不等于waiting time不为0的车辆数量
        # 车道的停车数量是单独计算的，自适应计算的。
        lane_attr_value = defaultdict(float)
        for lane in self.upstream_lanes+self.downstream_lanes:
            lane_attr_value[lane] = self.eng.lane.getLastStepHaltingNumber(lane)
            '''
            logging.info(f"phase: {self.eng.trafficlight.getRedYellowGreenState(self.id).lower()}")
            logging.info(f"{[self.lane_2_updownstream[lane]['from'] for lane in self.traffic_light_lanes]}")
            logging.info(f"{lane},{[i for i,x in enumerate([self.lane_2_updownstream[lane]['from'] for lane in self.traffic_light_lanes]) if x == lane]},{[self.eng.trafficlight.getRedYellowGreenState(self.id).lower()[ind] for ind in [i for i,x in enumerate([self.lane_2_updownstream[lane]['from'] for lane in self.traffic_light_lanes]) if x == lane]]} compare halting nums:{lane_attr_value[lane]}, \
                         {len([veh for veh in self.eng.lane.getLastStepVehicleIDs(lane) if self.eng.vehicle.getWaitingTime(veh) != 0])}, \
                        '\n',{[(veh,self.eng.vehicle.getWaitingTime(veh)) for veh in self.eng.lane.getLastStepVehicleIDs(lane)]},'\n', \
                            {[self.vehicles[veh].AccumulatedWaitingTime for veh in self.eng.lane.getLastStepVehicleIDs(lane)]},'\n',\
                        {[veh for veh in self.eng.lane.getLastStepVehicleIDs(lane) if self.eng.vehicle.getWaitingTime(veh) != 0]},'\n', \
                            {self.eng.lane.getLastStepVehicleIDs(lane)}")
            '''
        return lane_attr_value
    
    def get_lane_waiting_time(self):
        # 车道的waiting time等于车辆的waiting time之和
        # 每次灯切换都会导致启动的车辆的waiting time清零
        lane_attr_value = defaultdict(float)
        for lane in self.upstream_lanes+self.downstream_lanes:
            lane_attr_value[lane] = self.eng.lane.getWaitingTime(lane)
            '''
            logging.info(f"phase: {self.eng.trafficlight.getRedYellowGreenState(self.id).lower()}")
            #logging.info(f"{[self.lane_2_updownstream[lane]['from'] for lane in self.traffic_light_lanes]}")
            logging.info(f"{lane},{[i for i,x in enumerate([self.lane_2_updownstream[lane]['from'] for lane in self.traffic_light_lanes]) if x == lane]},{[self.eng.trafficlight.getRedYellowGreenState(self.id).lower()[ind] for ind in [i for i,x in enumerate([self.lane_2_updownstream[lane]['from'] for lane in self.traffic_light_lanes]) if x == lane]]}compare waiting times:{lane_attr_value[lane]}, \
                         {sum([self.vehicles[veh].AccumulatedWaitingTime for veh in self.eng.lane.getLastStepVehicleIDs(lane)])}, \
                        '\n',{[(veh,self.eng.vehicle.getWaitingTime(veh)) for veh in self.eng.lane.getLastStepVehicleIDs(lane)]},'\n', \
                            {[self.vehicles[veh].AccumulatedWaitingTime for veh in self.eng.lane.getLastStepVehicleIDs(lane)]},'\n',\
                            {self.eng.lane.getLastStepVehicleIDs(lane)}")
            '''
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
        for lane in self.upstream_lanes:
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
        
        vehicles = []
        for lane in self.upstream_lanes:
            lane_v = list(self.eng.lane.getLastStepVehicleIDs(lane))
            vehicles.extend(lane_v)
            '''
            logging.info(f"""travel time: {lane},
{self.eng.lane.getTraveltime(lane)}
{self.eng.lane.getLastStepHaltingNumber(lane)}
{self.eng.lane.getLastStepVehicleNumber(lane)}
{self.eng.lane.getWaitingTime(lane)}
{lane},
{[i for i,x in enumerate([self.lane_2_updownstream[lane]['from'] for lane in self.traffic_light_lanes]) if x == lane]}
{[self.eng.trafficlight.getRedYellowGreenState(self.id).lower()[ind] for ind in [i for i,x in enumerate([self.lane_2_updownstream[lane]['from'] for lane in self.traffic_light_lanes]) if x == lane]]}
""")
'''
            #for vv in lane_v:
            #    logging.info(f"{lane},{self.vehicles[vv].AccumulatedWaitingTime}")
        total_vehicles = list(set(vehicles) | set(self.last_step_vehicles))
        getin_vehicles = list(set(vehicles) - set(self.last_step_vehicles))
        leaved_vehicles = list(set(self.last_step_vehicles) - set(vehicles))
        
        vehicles_dec = len(vehicles) - len(self.last_step_vehicles)
        
        # 总延迟，并非当前动作的结果，是累积动作的结果
        for veh in leaved_vehicles:
            self.leaved_delay += self.vehicles[veh].AccumulatedWaitingTime
            self.leaved_vehicles += 1
        if self.leaved_vehicles != 0:
            average_delay = self.leaved_delay/self.leaved_vehicles
        else:
            average_delay = 0
        
        # 当前动作导致多少车辆通过交叉口
        throughput = len(leaved_vehicles)
        
        # 当前动作导致车辆等待时间的增长情况
        thisstep_total = 0
        for veh in vehicles:
            thisstep_total += self.vehicles[veh].AccumulatedWaitingTime
        wait_time_ascend = thisstep_total - self.last_step_waittime
        
        # 当前动作导致等车数量的增长情况
        total_wait_nums = 0
        for lane in self.upstream_lanes:
            total_wait_nums += self.eng.lane.getLastStepHaltingNumber(lane)
        waitnums_asc = total_wait_nums - self.last_step_watinums
        last_step_watinums = self.last_step_watinums
        
        # 

        self.last_step_vehicles = vehicles
        self.last_step_waittime = thisstep_total
        self.last_step_watinums = total_wait_nums
        
        return Indicators(total_vehicles = len(vehicles),
                          wait_time_ascend = wait_time_ascend,
                          throughput = throughput,
                          average_delay = average_delay,
                          total_wait_nums = last_step_watinums,
                          waitnums_asc = waitnums_asc,
                          vehicles_dec = vehicles_dec)
    #endregion
    
    
    #region reward
    def get_reward(self):
        indicator = self.get_all_info()
        #reward = ( -indicator.waitnums_asc + indicator.throughput)/(indicator.total_vehicles+1)  
        reward = - indicator.total_wait_nums/(indicator.total_vehicles+1)
        # indicator.throughput - 1 if indicator.total_wait_nums > 0 else 0.01
        #- indicator.total_wait_nums/(indicator.total_vehicles+1)
        return reward, indicator
    #endregion
    
    
if __name__ == '__main__':
    True
