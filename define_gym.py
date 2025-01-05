import gym
from gym import spaces

class SimpleCounterEnv(gym.Env):
    def __init__(self):
        super(SimpleCounterEnv, self).__init__()
        
        # 定义动作空间：0表示增加，1表示减少
        self.action_space = spaces.Discrete(2)
        
        # 定义观察空间：计数器的值范围为0到10
        self.observation_space = spaces.Box(low=0, high=10, shape=(1,), dtype=int)
        
        # 初始化计数器
        self.state = 5  # 初始值为5
        self.done = False

    def reset(self):
        # 重置环境状态
        self.state = 5
        self.done = False
        return self.state

    def step(self, action):
        # 根据动作更新状态
        if action == 0:  # 增加
            self.state = min(self.state + 1, 10)
        elif action == 1:  # 减少
            self.state = max(self.state - 1, 0)

        # 检查是否结束
        if self.state == 10 or self.state == 0:
            self.done = True

        # 返回状态、奖励、是否结束和额外信息
        reward = 1 if self.state < 10 else 0  # 计数器未到达10时给予奖励
        return self.state, reward, self.done, {}

    def render(self, mode='human'):
        # 可视化环境状态
        print(f"Current State: {self.state}")

# 注册环境
gym.envs.registration.register(
    id='SimpleCounter-v0',
    entry_point='__main__:SimpleCounterEnv',
)

# 测试自定义环境
if __name__ == "__main__":
    env = gym.make('SimpleCounter-v0')
    state = env.reset()
    done = False

    while not done:
        action = env.action_space.sample
