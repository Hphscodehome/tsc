import traci
import logging
# define_intersection_class.py
# test_traci.py
from utils.position import calculate_distance
# 启动仿真
state = ''.join(['r' for _ in range(16)])
logging.warning(f"state")
logging.warning(f"state:{state}")
traci.start(["sumo", "-c", "/data/hupenghui/TSC/data/syn1_1x1_1h/data.sumocfg"])
intersection = 'intersection_1_1'
lane_id = 'road_1_2_3_1'
traci.trafficlight.setRedYellowGreenState(intersection,state)
logging.warning(traci.trafficlight.getRedYellowGreenState(intersection))
# 运行仿真
time = 300
while time > 0:
    traci.simulationStep()
    logging.warning(traci.trafficlight.getRedYellowGreenState(intersection))
    lane_vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
    all = []
    for veh in lane_vehicles:
        pos = traci.vehicle.getLanePosition(veh)
        posxy = traci.vehicle.getPosition(veh)
        speed = traci.vehicle.getSpeed(veh)
        distance = calculate_distance(posxy,(800.0, 600.0))
        all.append((veh,
                    pos, 
                    distance, 
                    posxy,
                    speed,
                    traci.vehicle.getTimeLoss(veh),
                    traci.vehicle.getWaitingTime(veh),
                    traci.vehicle.getAccumulatedWaitingTime(veh),
                    traci.vehicle.getDepartDelay(veh)
                    ))
    logging.warning(f"{all}")
    '''
    for lane_id in lines:
        if ':' not in lane_id:
            lane_length = traci.lane.getLength(lane_id)
            queue_length = traci.lane.getLastStepVehicleNumber(lane_id)
            average_speed = traci.lane.getLastStepMeanSpeed(lane_id)
            logging.warning(lane_length, 
                  queue_length, 
                  average_speed, 
                  traci.lane.getAllowed(lane_id),
                  traci.lane.getLastStepLength(lane_id),#车辆的平均长度，5米
                  traci.lane.getLastStepVehicleNumber(lane_id),
                  traci.lane.getWaitingTime(lane_id),
                  traci.lane.getPendingVehicles(lane_id),
                  traci.lane.getLastStepHaltingNumber(lane_id),
                  traci.lane.getLastStepVehicleIDs(lane_id))
            break
    logging.warning("排队车辆数量:", queue_length)
    logging.warning("平均车辆速度:", average_speed)
    t = [traci.trafficlight.getAllProgramLogics(traid),
         traci.trafficlight.getControlledLanes(traid),
                    traci.trafficlight.getControlledLinks(traid),
                    traci.trafficlight.getPhase(traid),
                    traci.trafficlight.getPhaseDuration(traid),
                    traci.trafficlight.getPhaseName(traid),
                    traci.trafficlight.getProgram(traid),
                    traci.trafficlight.getRedYellowGreenState(traid)]
    logging.warning(f"{t}")
    '''
    time -= 1

# 关闭仿真
traci.close()
