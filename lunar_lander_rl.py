import copy
import os.path
import random
import shutil
from collections import deque
import matplotlib.pyplot as plt
import gymnasium as gym
import sys
import pygame
import torch
import torch.nn as nn
import numpy as np
import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class QNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, state):
        return self.net(state)


class Agent:
    def __init__(self, env):
        self.method = 'none'
        self.action_space = env.action_space
        self.state_space = env.observation_space

        self.gamma = 0.98
        self.epsilon = 0.2

    def save_weights(self):
        pass

    def experience_store(self, experience):
        pass

    def train_net(self):
        pass

    def explore(self, state):
        return self.action_space.sample()

    def take_action(self,state):
        return self.action_space.sample()

    def test(self, env):
        state, info = env.reset()
        action = self.take_action(state)
        ep_reward = 0

        while True:
            state_nx, reward, terminated, truncated, info = env.step(action)
            action_nx = self.take_action(state_nx)
            ep_reward += reward

            state = state_nx
            action = action_nx

            if terminated or truncated:
                break
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    sys.exit()
        print(f"Test: episode reward = {ep_reward}")


class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)


def main():
    EPISODE_NUM = 0
    MAX_EPISODE_STEPS = 2000
    # ENV_NAME = 'Pendulum-v1'
    ENV_NAME = 'LunarLanderContinuous-v3'
    # ENV_NAME = 'MountainCarContinuous-v0'

    env = gym.make(ENV_NAME, render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)

    # agent = DQN(env, weights_path='./weights/dqn_cartpole.pth')
    agent = Agent(env)
    ep_return_list = []
    exp_drop_cnt = 0

    # 采样
    for episode_i in range(EPISODE_NUM):
        state, info = env.reset()
        action = agent.explore(state)
        ep_reward = 0

        for _ in range(MAX_EPISODE_STEPS):
            state_nx, reward, terminated, truncated, info = env.step(action)
            action_nx = agent.explore(state_nx)
            ep_reward += reward

            agent.experience_store([state, action, reward, state_nx, action_nx, terminated])

            if agent.method == 'DQN':
                agent.train_net()

            state = state_nx
            action = action_nx

            if terminated or truncated:
                break

        if agent.method == 'REINFORCE':
            agent.train_net()

        agent.epsilon = max(0.1, agent.epsilon * 0.995)
        ep_return_list.append(ep_reward)
        print(f"Episode {episode_i}: episode reward = {ep_reward}, epsilon = {agent.epsilon:.4f}, exp dropped = {exp_drop_cnt}")

    agent.save_weights()

    env.close()
    plt.plot(ep_return_list)
    plt.show()

    # 测试
    for _ in range(10):
        agent.test(gym.make(ENV_NAME, render_mode='human', max_episode_steps=MAX_EPISODE_STEPS))


if __name__ == '__main__':
    main()
