# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 21:45:43 2025

@author: gaffliu
"""

"""
!apt update
!apt install python-opengl xvfb -y
!pip install gym[box2d]==0.18.3 pyvirtualdisplay tqdm numpy==1.19.5 torch==1.8.1
"""

"""
pip install wheel  # 确保wheel已安装

# 安装PyOpenGL和依赖
pip install PyOpenGL PyOpenGL-accelerate

pip install gym

#下载swig程序 https://www.swig.org/download.html
#解压缩设置path环境变量
#pip install box2d-py  # 替代官方的Box2D，兼容性更好

pip install gymnasium[box2d]

pip install pyvirtualdisplay tqdm numpy

"""

#Next, set up virtual display，and import all necessaary packages.

"""
打开Spyder，进入 Tools > Preferences > IPython Console > Graphics。
将 Backend 改为 Automatic 或 Qt5。
"""



import gym
import matplotlib.pyplot as plt

from IPython import display

import random  # 添加此行
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm  # 如果Spyder不支持tqdm.notebook，改用普通tqdm
import os


"""
# 在创建gym环境时指定渲染模式 测试

env = gym.make("LunarLander-v2", render_mode="human")
observation, _ = env.reset()  # reset() 返回 (observation, info)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated  # 合并终止和截断状态
    if done:
        break
env.close()
"""

#Make your HW result to be reproducible.

seed = 543 # Do not change this
def fix(env, seed):
  env.reset(seed=seed)
  env.action_space.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.use_deterministic_algorithms(True)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

"""
Last, call gym and build an Lunar Lander environment.
"""

env = gym.make('LunarLander-v2',render_mode="rgb_array")
fix(env, seed) # fix the environment Do not revise this !!!

"""
What Lunar Lander？
“LunarLander-v2”is to simulate the situation when the craft lands on the surface of the moon.

This task is to enable the craft to land "safely" at the pad between the two yellow flags.

Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector.



"LunarLander-v2" actually includes "Agent" and "Environment".

In this homework, we will utilize the function step() to control the action of "Agent".

Then step() will return the observation/state and reward given by the "Environment".
"""


"""
Observation / State
First, we can take a look at what an Observation / State looks like.
# 状态空间说明：
# 8维连续状态向量包含：
# 0. x坐标（水平位置）
# 1. y坐标（垂直高度）
# 2. x方向速度
# 3. y方向速度
# 4. 角度（0为垂直，正值为顺时针）
# 5. 角速度 
# 6. 左腿触地标志（0/1）
# 7. 右腿触地标志（0/1)
#输出：Box([-inf -inf -inf ...], [inf inf inf ...], (8,), float32)
"""


print(env.observation_space) 

"""
Action
Actions can be taken by looks like

# 输出：Discrete(4)
# 动作空间详解：
# 离散型动作空间（4种选择）：
# 0: 不执行任何推进（关闭所有引擎）
# 1: 启动左侧引擎（向右推进）
# 2: 启动主底部引擎（向上推进） 
# 3: 启动右侧引擎（向左推进）

"""

print(env.action_space)

"""
Discrete(4) implies that there are four kinds of actions can be taken by agent.

0 implies the agent will not take any actions
2 implies the agent will accelerate downward
1, 3 implies the agent will accelerate left and right
Next, we will try to make the agent interact with the environment. Before taking any actions, we recommend to call reset() function to reset the environment. Also, this function will return the initial state of the environment.
"""

# 环境初始化：
# reset() 作用：
# 1. 重置环境到初始状态（随机或固定位置，取决于seed设置）
# 2. 返回初始状态向量（8维数组）

initial_state, _ = env.reset()
print(initial_state)

#Then, we try to get a random action from the agent's action space.
# 动作采样：
# action_space.sample() 随机选择动作（此处用于演示）
# 实际训练中应由策略网络生成动作
random_action = env.action_space.sample()
print(random_action)


"""
More, we can utilize step() to make agent act according to the randomly-selected random_action. The step() function will return four values:

# 交互过程：
# step(action) 返回：
# - observation: 新的状态（8维）
# - reward: 当前奖励值（如示例可能输出：-0.5）
# - terminated: 是否成功/失败终止（如坠毁或着陆）
# - truncated: 是否因步数限制终止（默认1000步）
# - info: 附加信息（空字典）

# done 合并两种终止条件：
# 当飞船成功着陆、坠毁或超时，done变为True

"""

observation, reward, terminated, truncated, info = env.step(random_action)

done = terminated or truncated  # 合并终止和截断状态
print(done)
     

"""
Reward
Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points.
# 奖励机制详解：
# 基础奖励 = 位移奖励 + 速度奖励 + 角度奖励
# 附加奖励：
# - 成功着陆：+100~140（根据着陆质量）
# - 安全着陆：+100（最终奖励）
# - 坠毁：-100（最终奖励）
# - 腿部触地：每条腿+10
# - 引擎使用：主引擎每帧-0.3
# 达标条件：单局累计奖励≥200

# 示例中-0.3表示：
# 当前帧使用了主引擎（-0.3），没有其他奖励

"""
print(reward)

"""
Random Agent
In the end, before we start training, we can see whether a random agent can successfully land the moon or not.
"""
  
env.reset()

# 初始化图像显示
img = plt.imshow(env.render())
plt.axis('off')

done = False
while not done:
    # 随机采样动作（0-3之间的整数）
    action = env.action_space.sample()
    # 执行动作并获取反馈
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # 更新图像
    img.set_data(env.render())
    display.display(plt.gcf())
    display.clear_output(wait=True)

env.close()

"""
Policy Gradient
Now, we can build a simple policy network. The network will return one of action in the action space.
"""
#策略梯度网络
class PolicyGradientNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, state):
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(self.fc2(hid))
        return F.softmax(self.fc3(hid), dim=-1)
    
#定义一个Critic网络来估计状态的价值
class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)  # 输出状态价值
        
    def forward(self, state):
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(self.fc2(hid))
        return self.fc3(hid)    

"""
Then, we need to build a simple agent. The agent will acts according to the output of the policy network above. There are a few things can be done by agent:

learn()：update the policy network from log probabilities and rewards.
sample()：After receiving observation from the environment, utilize policy network to tell which action to take. The return values of this function includes action and log probabilities.

"""
from torch.optim.lr_scheduler import StepLR
 
class ActorCriticAgent:
    def __init__(self, actor_network, critic_network):
        self.actor_network = actor_network
        self.critic_network = critic_network
        # 为Actor网络创建Adam优化器，学习率0.001（管理策略参数更新）
        self.actor_optimizer = optim.Adam(actor_network.parameters(), lr=0.001)
        # 为Critic网络创建Adam优化器，学习率0.001（管理价值参数更新）
        self.critic_optimizer = optim.Adam(critic_network.parameters(), lr=0.001)
        
    # 根据当前策略采样动作
    def sample(self, state):
        state_tensor = torch.FloatTensor(state)  # 将numpy状态转为PyTorch张量
        action_prob = self.actor_network(state_tensor)  # 通过Actor网络获得动作概率分布
        action_dist = Categorical(action_prob)  # 创建分类分布（用于采样动作）
        action = action_dist.sample()  # 根据概率分布采样具体动作
        log_prob = action_dist.log_prob(action)  # 计算该动作的log概率（用于梯度计算）
        return action.item(), log_prob  # 返回动作（int类型）和对应的log概率
    
    # 学习函数，更新两个网络的参数
    def learn(self, states, log_probs, rewards, next_states, dones, gamma=0.99):
        # 转换为Tensor
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        # 计算当前状态的价值
        current_values = self.critic_network(states).squeeze()
        
        # 计算下一个状态的价值（终止状态为0）
        next_values = self.critic_network(next_states).squeeze()
        next_values = next_values * (1 - dones)
        
        # 计算TD目标（即时奖励 + 折现后的下一状态价值）
        td_targets = rewards + gamma * next_values.detach()
        # 计算Critic的损失（预测值与TD目标的MSE）
        critic_loss = F.mse_loss(current_values, td_targets)
        
        # 计算优势函数（TD目标 - 当前价值，评估动作好坏）
        advantages = td_targets - current_values.detach()
        
        # 计算Actor的损失（最大化优势加权的log概率）
        log_probs_tensor = torch.stack(log_probs)
        actor_loss = -(log_probs_tensor * advantages).sum()
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def train_mode(self):
        self.actor_network.train()
        self.critic_network.train()
    
    def eval_mode(self):
        self.actor_network.eval()
        self.critic_network.eval()
        
#Lastly, build a network and agent to start training

actor_network = PolicyGradientNetwork()
critic_network = ValueNetwork()
agent = ActorCriticAgent(actor_network, critic_network)

"""
Training Agent
Now let's start to train our agent. Through taking all the interactions between agent and environment as training data, the policy network can learn from all these attempts,
"""
agent.train_mode()  # Switch network into training mode 
EPISODE_PER_BATCH = 5  # 每个批次收集5个episode的数据
NUM_BATCH = 500        # totally update the agent for 400 time

# 存储每批次的平均总奖励、平均最终奖励
avg_total_rewards, avg_final_rewards = [], []

prg_bar = tqdm(range(NUM_BATCH))
for batch in prg_bar:
    
    # 存储每个动作的log概率、存储即时奖励
    log_probs, rewards = [], []
    # 存储状态转移序列（Actor-Critic新增）
    states, next_states, dones = [], [], []

    # collect trajectory
    for episode in range(EPISODE_PER_BATCH):
        
        state, _ = env.reset()  # 正确获取状态数组
        # 单个episode循环
        while True:
            states.append(state.copy())
            # 采样动作（包含探索）
            action, log_prob = agent.sample(state) # at, log(at|st)
            # 与环境交互
            next_state, reward,  terminated, truncated, info = env.step(action)
            done = terminated or truncated  # 合并终止状态
            
            # 保存动作的log概率（用于梯度计算）
            log_probs.append(log_prob) # [log(a1|s1), log(a2|s2), ...., log(at|st)]

            # 保存动作的reward
            rewards.append(reward) # change here
            next_states.append(next_state.copy())
            dones.append(done)
            state = next_state
            # ! IMPORTANT !
            # Current reward implementation: immediate reward,  given action_list : a1, a2, a3 ......
            #                                                         rewards :     r1, r2 ,r3 ......
            # medium：change "rewards" to accumulative decaying reward, given action_list : a1,                           a2,                           a3, ......
            #                                                           rewards :           r1+0.99*r2+0.99^2*r3+......, r2+0.99*r3+0.99^2*r4+...... ,  r3+0.99*r4+0.99^2*r5+ ......
            # boss : implement Actor-Critic
            if done:
                break

    # 修改后的打印语句
    print(f"rewards looks like (length): {len(rewards)}")  
    if len(log_probs) > 0:
        print(f"log_probs element shape: {log_probs[0].detach().shape}, count: {len(log_probs)}")
    else:
        print("log_probs is empty")

    # 策略梯度更新（核心）
    agent.learn(states, log_probs, rewards, next_states, dones)
    print(f"log_probs looks like ", log_probs[0].detach().shape)  # 使用PyTorch的shape属性查看单个log_prob的形状
    print(f"Number of log_probs elements: {len(log_probs)}")       # 使用Python的len函数查看列表长度"""
    
"""    
Training Result
During the training process, we recorded avg_total_reward, which represents the average total reward of episodes before updating the policy network.

Theoretically, if the agent becomes better, the avg_total_reward will increase. The visualization of the training process is shown below:
"""

plt.plot(avg_total_rewards)
plt.title("Total Rewards")
plt.show()

"""
In addition, avg_final_reward represents average final rewards of episodes. To be specific, final rewards is the last reward received in one episode, indicating whether the craft lands successfully or not.

"""
plt.plot(avg_final_rewards)
plt.title("Final Rewards")
plt.show()

#save model
torch.save(agent.actor_network.state_dict(), "actor_model.ckpt") # only save best to prevent output memory exceed error
torch.save(agent.critic_network.state_dict(), "critic_model.ckpt") # only save best to prevent output memory exceed error

"""
Testing
The testing result will be the average reward of 5 testing
"""
fix(env, seed)
agent.eval_mode()  # set the network into evaluation mode
NUM_OF_TEST = 5 # Do not revise this !!!
test_total_reward = []
action_list = []
for i in range(NUM_OF_TEST):
  actions = []
  state, _ = env.reset()  # 正确获取状态数组

  img = plt.imshow(env.render())

  total_reward = 0

  done = False
  while not done:
      action, _ = agent.sample(state)
      actions.append(action)
      # 修改后
      state, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated

      total_reward += reward
      # 实时更新渲染（Jupyter适用）
      img.set_data(env.render())
      display.display(plt.gcf())
      display.clear_output(wait=True)
      
  print(total_reward)
  test_total_reward.append(total_reward)

  action_list.append(actions) # save the result of testing 


print(np.mean(test_total_reward))

#Action list


print("Action list looks like ", action_list)
print("Action list structure:")
for i, actions in enumerate(action_list):
    print(f"Test {i+1} actions count: {len(actions)}")


#Analysis of actions taken by agent

distribution = {}
for actions in action_list:
  for action in actions:
    if action not in distribution.keys():
      distribution[action] = 1
    else:
      distribution[action] += 1
print(distribution)

#Saving the result of Model Testing

PATH = "Action_List.npy"
np.save(PATH, np.array(action_list, dtype=object))  # 添加 dtype=object 参数
   
"""
This is the file you need to submit !!!
Download the testing result to your device

"""

#from google.colab import files
#files.download(PATH)

    
"""
Server
The code below simulate the environment on the judge server. Can be used for testing.
"""
action_list = np.load(PATH,allow_pickle=True) # The action list you upload
seed = 543 # Do not revise this
fix(env, seed)

agent.eval_mode()  # set network to evaluation mode

test_total_reward = []
if len(action_list) != 5:
  print("Wrong format of file !!!")
  exit(0)
for actions in action_list:
  state, _ = env.reset()  # 正确获取状态数组
  img = plt.imshow(env.render())

  total_reward = 0

  done = False

  for action in actions:
      state, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated  
      
      total_reward += reward
      if done:
        break

  print(f"Your reward is : %.2f"%total_reward)
  test_total_reward.append(total_reward)

#Your score

print(f"Your final reward is : %.2f"%np.mean(test_total_reward))

  