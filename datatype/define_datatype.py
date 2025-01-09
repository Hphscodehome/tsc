#region other-package
from pydantic import BaseModel
from typing import List,Dict,Tuple
#endregion

class Phase(BaseModel):
    phase_id: int
    phase_str: str
    phase_duration: float
    
class NetConfig(BaseModel):
    lane_2_shape: Dict[str,List[Tuple[float,float]]]
    intersection_2_updownstream: Dict[str,Dict[str,set]]
    lane_2_updownstream: Dict[str,Dict[str,str]]
    intersection_2_position: Dict[str,Tuple[float,float]]

class Vehicle(BaseModel):
    AccumulatedWaitingTime: float = 0.0
    
class Indicators(BaseModel):
    throughput: int
    average_delay: float

if __name__ == '__main__':
    # 测试 Vehicle 模型
    vehicle = Vehicle()
    print("Vehicle AccumulatedWaitingTime:", vehicle.AccumulatedWaitingTime)

    # 测试 Phase 模型
    phase = Phase(phase_id=1, phase_str="Green", phase_duration=30.0)
    print("Phase:", phase)

    # 测试 NetConfig 模型
    net_config = NetConfig(
        lane_2_shape={
            "lane_1": [(0.0, 0.0), (1.0, 1.0)],
            "lane_2": [(1.0, 1.0), (2.0, 2.0)]
        },
        intersection_2_updownstream={
            "intersection_1": {"upstream": {"lane_1"}, "downstream": {"lane_2"}},
        },
        lane_2_updownstream={
            "lane_1": {"upstream": "intersection_1", "downstream": "lane_2"},
        },
        intersection_2_position={
            "intersection_1": (1.0, 1.0),
        }
    )
    print("NetConfig:", net_config)

    # 测试 Indicators 模型
    indicators = Indicators(throughput=100, average_delay=5.5)
    print("Indicators:", indicators)