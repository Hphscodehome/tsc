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



'''
def Phase(*args):
    phase_id: int = args[0]
    phase_str: str = args[1]
    phase_duration: float = args[2]
    return {'phase_id':phase_id,'phase_str':phase_str,'phase_duration':phase_duration}
'''

if __name__ == '__main__':
    item = Vehicle()
    print(item.AccumulatedWaitingTime)