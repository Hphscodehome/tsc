`other-package` : 引入的公开包
`my-package` ： 自定义包

Intersection:
    observation: Dict['lane','mask']
    reward: float

World:
    observation: Dict[Intersection.observation]
    dones: Dict[bool]
    rewards: Dict[float]

World_actors:
    obs: Dict[Intersection.observation]
    actions: Dict[torch.tensor]
    
