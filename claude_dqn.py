import copy
import random
import collections

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import wandb



wandb.login()

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    def __init__(self, batch_size = 128, max_size = 1e4):
        self.batch_size = batch_size
        self.max_size = max_size
        self.buffer = collections.deque(maxlen = int(self.max_size))

    def add(self, observation, action, reward, done, next_observation):
        transition = (observation, action, reward, done, next_observation)
        self.buffer.append(transition)

    def can_sample(self):
        return len(self.buffer) >= self.batch_size
    
    def sample(self):
        transitions = random.sample(self.buffer, self.batch_size)
        batch = list(zip(*transitions))
        return batch
    
def gather(values, indices):
    one_hot = 1 
    

class Agent:
    def __init__(self, num_actions, environment, device, learning_rate = 3e-4, batch_size = 32, max_size = 1e4, eps_decay = 0.999, eps_min = 0.05):
        self.environment = environment
        self.device = device
        self.num_actions = num_actions
        self.buffer = ReplayBuffer(batch_size=batch_size, max_size = max_size)
        self.q_network = QNetwork(env = environment).to(device)
        self.target_network = copy.deepcopy(self.q_network)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr = learning_rate)
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.eps = 1.0

        self.training_steps = 0

    def select_action(self, observation):
        self.eps = max(self.eps_min, self.eps_decay*self.eps)

        if random.random() < self.eps:
            action = self.environment.action_space.sample()
        else:
            observation = torch.tensor(observation).to(self.device)
            q_values = self.q_network(observation)
            action = torch.argmax(q_values)
            action = action.item()

        return action
    
    def train(self):
        batch = self.buffer.sample()

        observations = torch.tensor(np.array(batch[0])).to(self.device)
        actions = torch.tensor(batch[1]).to(self.device)
        rewards = torch.tensor(batch[2]).to(self.device)
        dones = torch.tensor(batch[3]).to(self.device)
        next_observations = torch.tensor(np.array(batch[4])).to(device)

        next_q_values = self.target_network(next_observations)
        target_values = torch.max(next_q_values, dim = -1)[0]

        targets = rewards + 0.99 * (1-dones.to(torch.int)) * target_values

        q_values = self.q_network(observations)
        action_values = torch.gather(input = q_values, dim = 1, index = actions.unsqueeze(1))
        td_error = (targets - action_values)**2
        loss = torch.mean(td_error)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.training_steps % 25 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.training_steps +=1


env = gym.make("CartPole-v1")
device = ("cuda" if torch.cuda.is_available else "cpu")
num_actions = env.action_space.n
agent = Agent(num_actions, env, device, 3e-4, eps_decay=0.9999, eps_min=0.05, batch_size=128)

wandb.init()

time_steps = 0
for e in range(100000):
    observation, _ = env.reset()
    done = False
    episode_return = 0
    while not done:
        action = agent.select_action(observation)
        next_observation, reward, truncation, terminal, _ = env.step(action)

        done = truncation or terminal

        agent.buffer.add(observation, action, reward, done, next_observation)

        # Critical step that was left out in the video
        observation = next_observation

        episode_return += reward

        if agent.buffer.can_sample() and time_steps % 10 == 0:
            agent.train()

        time_steps += 1

    wandb.log({"episode_return": episode_return, "eps": agent.eps, "time_steps": time_steps})

print("Done")




