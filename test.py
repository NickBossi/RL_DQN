import random
import torch
import gymnasium as gym
import time
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1", render_mode = "human")

AB = ["A","B"]
now = list(set(AB) - set(["A"]))

print(random.choice(["A","B"]))
print(now)


env.reset()
for i in range(100000):
    env.render()
    action = env.action_space.sample()
    print(action)


    env.step(action)

'''
import gymnasium as gym

env = gym.make("CartPole-v1")

observation, _ = env.reset()
action = env.action_space.sample()

step = [next_observaton, rewards, terminated, truncated,_] = env.step(action)

print(step)
'''
things = [[1,2,3],[4,5,6],[7,8,9]]

sample = random.sample(things, 2)

thiiings = list(zip(*sample))
print(thiiings)




q_values = torch.tensor([[1,2,3,4],[5,6,7,8]])
actions = torch.tensor([[2],[3]])
action_values = torch.gather(input = q_values, dim = 1, index = actions)
print(action_values)

print(torch.cuda.is_available())