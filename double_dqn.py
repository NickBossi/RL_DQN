import copy
import random
import collections

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import wandb
import time



wandb.login()

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, env.action_space.n),
        )
        self.optimizer = optim.Adam(self.parameters(), lr = 3e-4)

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
    

class Agent:
    def __init__(self, num_actions, environment, device, learning_rate = 3e-4, batch_size = 32, max_size = 1e4, eps_decay = 0.99999, eps_min = 0.05):
        self.environment = environment
        self.device = device
        self.num_actions = num_actions
        self.buffer = ReplayBuffer(batch_size=batch_size, max_size = max_size)

        self.q_network_A = QNetwork(env = environment).to(device)
        self.q_network_B = copy.deepcopy(self.q_network_A).to(device)
        self.target_network = self.q_network_A
        self.q_network = self.q_network_B

        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.eps = 1.0
        self.target_weigting = 0.01
        self.choice = ""

        self.training_steps = 0

    def select_action(self, observation):
        self.eps = max(self.eps_min, self.eps_decay*self.eps)

        if random.random() < self.eps:
            action = self.environment.action_space.sample()
        else:
            self.choice = random.choice(["A","B"])

            observation = torch.tensor(observation).to(self.device)

            if self.choice == "A":
                self.target_network = self.q_network_A
                self.q_network = self.q_network_B

            else:
                self.target_network = self.q_network_B
                self.q_network = self.q_network_A

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


        # Only change between this and normal DQN. We choose action based on target network, but use value of Q network
        with torch.no_grad():
            next_q_values = self.target_network(next_observations)
            target_values = torch.max(next_q_values, dim = -1)[0]
            targets = rewards + 0.99 * (1-dones.to(torch.int)) * target_values

        q_values = self.q_network(observations)

        action_values = torch.gather(input = q_values, dim = 1, index = actions.unsqueeze(1))

        loss = F.mse_loss(action_values, targets)
        self.q_network.optimizer.zero_grad()
        loss.backward()
        self.q_network.optimizer.step()

        if self.training_steps % 25 == 0:

            self.target_network = random.choice(["A","B"])

            if self.choice == "A":
                self.target_network = self.q_network_A
                self.q_network = self.q_network_B

            else:
                self.target_network = self.q_network_B
                self.q_network = self.q_network_A     


            for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(
                    self.target_weigting*param.data + (1-self.target_weigting)*target_param.data
                    )


        self.training_steps +=1


env = gym.make("CartPole-v1")
device = ("cuda" if torch.cuda.is_available() else "cpu")
num_actions = env.action_space.n
agent = Agent(num_actions, env, device, 3e-4, eps_decay=0.9999, eps_min=0.05, batch_size=128)

wandb.init()

visualise = True

time_steps = 0
for e in range(50000):

    if e>1000 and visualise:
        env = gym.make("CartPole-v1", render_mode = "human")
        visualise = False
    
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




