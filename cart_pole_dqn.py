import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random
from collections import deque
import numpy as np

NUM_STEPS = 1000
NUM_EPISODES = 1000
REPLAY_BUFFER = deque(maxlen = 1000)
BATCH_SIZE = 16
DONE = False
DISCOUNT_FACTOR = 0.99


device = ("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = torch.nn.MSELoss()

env = gym.make("CartPole-v1")
observation, info = env.reset()

NUM_ACTIONS = env.action_space.n
EPSILON = 1/NUM_ACTIONS
SIZE_OBSERVATIONS = int(env.observation_space.shape[0])

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n)
        )
    
    def forward(self, input):
        return self.network(input)

NET = neural_net().to(device)
NET.train()
optimizer = optim.Adam(NET.parameters(), lr = 3e-4)




for i in range(NUM_EPISODES):
    epsiode_return = 0
    current_state,_ = env.reset()
    current_state = torch.tensor(current_state).to(device)
    action_values = NET(current_state)
    random_number = random.random()
    
    if random_number>EPSILON:
        action = int(torch.argmax(action_values))
    else:
        action = env.action_space.sample()
    
    next_state, reward, terminated, truncated, _ = env.step(action)

    epsiode_return += reward  
    REPLAY_BUFFER.append((current_state, action, reward, next_state))
    DONE = terminated or truncated

    current_state = next_state

    while not DONE:
        action_values = NET(torch.tensor(current_state).to(device))
        random_number = random.random()
    
        if random_number>EPSILON:
            action = int(torch.argmax(action_values))
        else:
            action = env.action_space.sample()
        
        next_state, reward, terminated, truncated, _ = env.step(action)

        DONE = terminated or truncated
        epsiode_return += reward 

        REPLAY_BUFFER.append((current_state, action, reward, next_state))

        current_state = next_state

        if (len(REPLAY_BUFFER)>BATCH_SIZE):
            total_loss = torch.tensor(0.0, requires_grad=True).to(device)
            batch = (random.sample(list(REPLAY_BUFFER), BATCH_SIZE))
            optimizer.zero_grad()
            #print(batch)
            for sample in batch:
                current_state, action, reward, next_state = sample
                current_state = torch.tensor(current_state).to(device)
                next_state = torch.tensor(next_state).to(device)
                action_value = NET(current_state)[action]
                target_action_value = torch.max(NET(next_state))

                target = reward + (1-DONE)*DISCOUNT_FACTOR*target_action_value

                loss = loss_fn(target, action_value)
                total_loss +=loss.item()

            total_loss.backward()
            optimizer.step()
    
    print(f"Episode {i} had a return of {epsiode_return}")
            

        

