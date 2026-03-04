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

    def take_action(self, state):
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


class DQN(Agent):
    def __init__(self, env, weights_path):
        super().__init__(env)
        self.method = 'DQN'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.hidden_dim = 64
        self.qnet = QNet(self.state_space.shape[0], self.hidden_dim, self.action_space.n).to(self.device)
        self.qnet_t = copy.deepcopy(self.qnet).to(self.device)
        self.batch_size = 256
        self.target_update_interval = 10
        self.target_update_cnt = 0
        self.alpha = 0.001
        self.gamma = 0.98
        self.epsilon = 0.5
        self.loss_fn = nn.MSELoss()

        self.experiences = deque(maxlen=self.batch_size * 100)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.alpha)

        self.weights_path = weights_path
        if weights_path is not None:
            if os.path.exists(weights_path):
                print(f"Loading weights from {weights_path}")
                weights = torch.load(weights_path)
                self.qnet.load_state_dict(weights)
                self.qnet_t.load_state_dict(weights)

        self.train_exp_interval = 5
        self.train_exp_cnt = 0

    def experience_store(self, experience):
        s, a, r, s_nx, a_nx, terminated = experience
        self.experiences.append((s, a, r, s_nx, terminated))
        return

    def train_net(self):
        self.train_exp_cnt += 1

        if self.train_exp_cnt >= self.train_exp_interval:
            if len(self.experiences) < self.batch_size:
                return
            self.train_exp_cnt = 0
            self.qnet.train()
            batch_s, batch_a, batch_r, batch_s_nx, batch_terminated = zip(*random.sample(self.experiences, self.batch_size))
            batch_s = torch.from_numpy(np.array(batch_s)).float().to(self.device)
            batch_a = torch.tensor(batch_a, dtype=torch.long).to(self.device)
            batch_r = torch.tensor(batch_r, dtype=torch.float).to(self.device)
            batch_s_nx = torch.tensor(batch_s_nx, dtype=torch.float).to(self.device)
            batch_terminated = torch.tensor(batch_terminated, dtype=torch.float).to(self.device)

            # 当前的Q值，用在线网络
            qsa = self.qnet(batch_s).gather(1, batch_a.unsqueeze(1)).squeeze(1)
            # 下一步的最大值，用目标网络
            q_nx_max = torch.max(self.qnet_t(batch_s_nx).detach(), dim=1)[0]
            loss = self.loss_fn(qsa, batch_r + self.gamma * q_nx_max * (1-batch_terminated))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.target_update_cnt += 1
            if self.target_update_cnt % self.target_update_interval == 0:
                self.qnet_t.load_state_dict(self.qnet.state_dict())

    def explore(self, state):
        if random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return self.take_action(state)

    def take_action(self, state):
        self.qnet.eval()
        with torch.no_grad():
            qs = self.qnet(torch.tensor(state).unsqueeze(0).to(self.device))
            best_action = torch.argmax(qs, dim=1).item()
        return best_action

    def save_weights(self):
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        if os.path.exists(self.weights_path):
            shutil.copy(self.weights_path, self.weights_path + '.backup' + f'{int(time.time())}')
        torch.save(self.qnet.state_dict(), self.weights_path)


class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)


class Reinforce(Agent):
    def __init__(self, env):
        super().__init__(env)
        self.method = 'REINFORCE'
        self.experiences = list()
        self.gamma = 0.98
        self.epsilon = 0.2

        # policy network
        self.batch_size = 64
        self.epoch_num = 2
        self.qsa_exp = list()
        self.hidden_dim = 64
        self.alpha = 0.01
        self.pn = PolicyNet(self.state_space.shape[0], self.hidden_dim, self.action_space.n)
        self.optimizer = torch.optim.Adam(self.pn.parameters(), lr=self.alpha)

    def experience_store(self, experience):
        self.experiences.append(experience)

    def train_net(self):
        g = 0

        # 估计动作值, every visit
        for exp in reversed(self.experiences):
            s, a, r, s_nx, *_ = exp
            g = r + self.gamma * g
            self.qsa_exp.append((s, a, g))
        self.experiences.clear()

        if len(self.qsa_exp) >= self.batch_size * self.epoch_num:
            self.pn.train()

            batch_s, batch_a, batch_g = zip(*self.qsa_exp)
            batch_s = torch.tensor(batch_s, dtype=torch.float)
            batch_a = torch.tensor(batch_a, dtype=torch.long).unsqueeze(1)
            batch_q = torch.tensor(batch_g, dtype=torch.float).unsqueeze(1)
            # batch_q -= batch_q.mean()  # baseline
            # 优化策略网络
            ln_pi = torch.log(self.pn(batch_s)).gather(1, batch_a)
            loss = -(ln_pi * batch_q).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.qsa_exp.clear()

        return

    def take_action(self, state):
        self.pn.eval()
        probs = self.pn(torch.tensor(state, dtype=torch.float).unsqueeze(0))

        return torch.multinomial(probs, 1).item()

    def explore(self, state):
        return self.take_action(state)

    def save_weights(self):
        return


def main():
    EPISODE_NUM = 1000
    MAX_EPISODE_STEPS = 2000

    env = gym.make('CartPole-v1', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)

    # agent = DQN(env, weights_path='./weights/dqn_cartpole.pth')
    agent = Reinforce(env)
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
                if len(agent.experiences) > agent.batch_size:
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
        agent.test(gym.make('CartPole-v1', render_mode='human', max_episode_steps=MAX_EPISODE_STEPS))


if __name__ == '__main__':
    main()
