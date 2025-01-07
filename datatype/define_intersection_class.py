#region other-package
import asyncio
from functools import partial
from collections import defaultdict
import numpy as np
#endregion

#region my-package
from datatype.define_datatype import Phase
from utils.position import judge_cross
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
    
    
    
    
    
    #region lane
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
    
    #region edge
    def get_edge_average_speed(self):
        pass
    
    def get_edge_waiting_time(self):
        pass
    #endregion
    
    #region phase_light
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
    
    #region vehicle
    def get_vehicle_map(self, option=0, average=True):
        """
        option: whether is the state of downstream/upstream/current
        """ 
        max_vehicle_num = 40
        num_lane = len(self.lanes)

        # 2 channel: position and speed
        if option == 0:
            vehicle_map = np.zeros((2, num_lane, max_vehicle_num))
            for i, lane in enumerate(self.lanes):
                # get lanes vehicles
                lane_vehicles = self.eng.lane.getLastStepVehicleIDs(lane)
                lane_length = self.lane_length[i]
                lane_max_speed = self.lane_max_speed[i]
                for veh in lane_vehicles:
                    pos = self.eng.vehicle.getLanePosition(veh)
                    speed = self.eng.vehicle.getSpeed(veh)
                    idx = int((lane_length - pos) // self.vehicle_gap)

                    if idx >= max_vehicle_num:
                        continue

                    vehicle_map[0, i, idx] += 1
                    vehicle_map[1, i, idx] += speed / lane_max_speed 
        else:
            vehicle_map = np.zeros((2, num_lane, 3, max_vehicle_num))
            if option == 1:
                all_lane_map, all_lane_length, all_lane_speed = self.downstream_map, self.downstream_length, self.downstream_speed
            else:
                all_lane_map, all_lane_length, all_lane_speed = self.upstream_map, self.upstream_length, self.upstream_speed
            for i, lane in enumerate(self.lanes):
                # get lanes vehicles
                for j, l in enumerate(all_lane_map.get(lane, [])):
                    
                    lane_vehicles = self.eng.lane.getLastStepVehicleIDs(l)
                    lane_length = all_lane_length[l]
                    lane_max_speed = all_lane_speed[l]
                    for veh in lane_vehicles:
                        pos = self.eng.vehicle.getLanePosition(veh)
                        if option != 1:
                            pos = lane_length - pos
                        speed = self.eng.vehicle.getSpeed(veh)
                        idx = int(pos // self.vehicle_gap)

                        if idx >= max_vehicle_num:
                            continue

                        vehicle_map[0, i, j, idx] += 1
                        vehicle_map[1, i, j, idx] += speed / lane_max_speed 
            if average:
                vehicle_map = np.mean(vehicle_map, axis=2) 

        return vehicle_map
    
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
    
    
    def get_current_available_lane(self):
        """current available lanes. e.g. [1, 0, 0, 1,..., 0], 1 means available
        """
        return self.remove_right_phase_mask[self.curr_action]

    def get_phase_mask(self):
        return self.remove_right_phase_mask

    def get_phase_available_lane(self, phase):
        return self.phase_available_lanes[phase]

    def get_neh_phase(self):
        """neh phase by one-hot
        """
        num_lane = len(self.lanes)
        neh_phase = np.ones((num_lane, 3))
        all_lane_map = self.upstream_map
        for i, lane in enumerate(self.lanes):
            # get lanes vehicles
            for j, l in enumerate(all_lane_map.get(lane, [])):
                inter, lane_idx = self.upstream_idx[l]
                inter_state = self.eng.trafficlight.getRedYellowGreenState(inter)
                lane_state = inter_state[lane_idx]
                if lane_state == 'r':
                    neh_phase[i, j] = 0

        return neh_phase

    def get_vehicle_map(self, option=0, average=True):
        """
        option: whether is the state of downstream/upstream/current
        """ 
        max_vehicle_num = 40
        num_lane = len(self.lanes)

        # 2 channel: position and speed
        if option == 0:
            vehicle_map = np.zeros((2, num_lane, max_vehicle_num))
            for i, lane in enumerate(self.lanes):
                # get lanes vehicles
                lane_vehicles = self.eng.lane.getLastStepVehicleIDs(lane)
                lane_length = self.lane_length[i]
                lane_max_speed = self.lane_max_speed[i]
                for veh in lane_vehicles:
                    pos = self.eng.vehicle.getLanePosition(veh)
                    speed = self.eng.vehicle.getSpeed(veh)
                    idx = int((lane_length - pos) // self.vehicle_gap)
                    if idx >= max_vehicle_num:
                        continue
                    vehicle_map[0, i, idx] += 1
                    vehicle_map[1, i, idx] += speed / lane_max_speed 
        else:
            vehicle_map = np.zeros((2, num_lane, 3, max_vehicle_num))
            if option == 1:
                all_lane_map, all_lane_length, all_lane_speed = self.downstream_map, self.downstream_length, self.downstream_speed
            else:
                all_lane_map, all_lane_length, all_lane_speed = self.upstream_map, self.upstream_length, self.upstream_speed
            for i, lane in enumerate(self.lanes):
                # get lanes vehicles
                for j, l in enumerate(all_lane_map.get(lane, [])):
                    
                    lane_vehicles = self.eng.lane.getLastStepVehicleIDs(l)
                    lane_length = all_lane_length[l]
                    lane_max_speed = all_lane_speed[l]
                    for veh in lane_vehicles:
                        pos = self.eng.vehicle.getLanePosition(veh)
                        if option != 1:
                            pos = lane_length - pos
                        speed = self.eng.vehicle.getSpeed(veh)
                        idx = int(pos // self.vehicle_gap)

                        if idx >= max_vehicle_num:
                            continue

                        vehicle_map[0, i, j, idx] += 1
                        vehicle_map[1, i, j, idx] += speed / lane_max_speed 
            if average:
                vehicle_map = np.mean(vehicle_map, axis=2) 

        return vehicle_map


    def get_queue_length_reward(self):
     
        return - np.mean([self.eng.lane.getLastStepHaltingNumber(lane) for lane in self.lanes if lane not in self.right_turn_lanes])

    def get_throughput_reward(self):
        """throughput in last action interval

        Returns:
            float: throughput reward
        """
        veh_list = []
        for lane in self.lanes:
            if lane not in self.right_turn_lanes:
                veh_list.extend(list(self.eng.lane.getLastStepVehicleIDs(lane)))

        cnt = 0
        for veh in veh_list:
            if veh in self.veh_list:
                cnt += 1

        throughput = (len(self.veh_list) - cnt)

        self.veh_list = veh_list

        return throughput / (len(self.lanes) - len(self.right_turn_lanes))
    
    def get_delay_reward(self):
        """time loss of vehicle in last action interval

        Returns:
            delay reward
        """

        max_vehicle_num = 40
        delay_reward = 0

        veh_time_loss_dict = {}

        for i, lane in enumerate(self.lanes):
            if lane in self.right_turn_lanes:
                continue
            # get lanes vehicles
            lane_vehicles = self.eng.lane.getLastStepVehicleIDs(lane)
            lane_length = self.lane_length[i]
            for veh in lane_vehicles:
                pos = self.eng.vehicle.getLanePosition(veh)
                idx = int((lane_length - pos) // self.vehicle_gap)

                # too far away vehicle, we do not consider
                if idx >= max_vehicle_num:
                    continue
            
                new_time_loss = self.eng.vehicle.getTimeLoss(veh)
                if veh in self.veh_time_loss_dict:
                    old_time_loss = self.veh_time_loss_dict[veh]
                    delay_reward += (new_time_loss - old_time_loss)
                
                veh_time_loss_dict[veh] = new_time_loss
        
        self.veh_time_loss_dict = veh_time_loss_dict

        return -delay_reward
        
if __name__ == '__main__':
    True
