import copy
import random

import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from examples import rl_utils


class DQN:
    def __init__(self, env: gym.Env):
        self.env = env
        self.exp_pool = []  # 经验池
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        )
        self.q_net = self.q_net.to(self.device)
        self.q_tar_net = copy.deepcopy(self.q_net)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=0.001)
        self.eps = 0.05
        self.gamma = 0.95
        self.batch_size = 128
        self.n_update = 8
        self.mem_max = 100000
        self.minimal_size = 600

        self.cnt = 0

    def take_action(self, state):
        # epsilon-greedy策略选取动作
        q_out = self.q_net(
            torch.tensor(state, dtype=torch.float32, device=self.device)
        )
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_out = self.q_net(state_tensor)
        if np.random.rand() < self.eps:
            action = self.env.action_space.sample()
        else:
            action = q_out.argmax(dim=1).item()
        return action

    def save_mem(self, transition):
        self.exp_pool.append(transition)
        if len(self.exp_pool) > self.mem_max:
            self.exp_pool.pop(0)

    def update(self):
        if len(self.exp_pool) >= self.minimal_size:
            exp_data = random.sample(self.exp_pool, self.batch_size)
            exp_data = list(zip(*exp_data))

            exp_state = torch.as_tensor(np.stack(exp_data[0]), dtype=torch.float32, device=self.device)
            exp_action = torch.as_tensor(np.stack(exp_data[1]), dtype=torch.int64, device=self.device).view(-1, 1)
            exp_reward = torch.as_tensor(np.stack(exp_data[2]), dtype=torch.float32, device=self.device)
            exp_state_nx = torch.as_tensor(np.stack(exp_data[3]), dtype=torch.float32, device=self.device)
            exp_done = torch.as_tensor(np.stack(exp_data[4]), dtype=torch.float32, device=self.device)

            q_out_all = self.q_net(exp_state).gather(1, exp_action).squeeze(1)
            q_max = self.q_tar_net(exp_state_nx).max(1)[0].detach()
            loss = torch.mean((exp_reward + self.gamma * q_max * (1-exp_done) - q_out_all)**2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 定期将评估网络参数复制到目标网络
            if self.cnt >= self.n_update:
                self.q_tar_net.load_state_dict(self.q_net.state_dict())
                self.cnt = 0
            else:
                self.cnt += 1


def main():
    lr = 1e-3
    num_episodes = 500
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env = gym.make('CartPole-v1', max_episode_steps=1000)
    state, info = env.reset(seed=0)
    agent = DQN(env)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                while True:
                    action = agent.take_action(state)
                    state_next, reward, done, truncated, info = env.step(action)
                    episode_return += reward
                    agent.save_mem((state, action, reward, state_next, done))
                    agent.update()
                    state = state_next
                    if done or truncated:
                        state, info = env.reset()
                        agent.cnt = 0
                        break
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format('car pole'))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format('car pole'))
    plt.show()

    torch.save(agent.q_net.state_dict(), "dqn_cartpole.pth")


def test():
    # 运行一局测试
    env = gym.make('CartPole-v1', render_mode="human", max_episode_steps=1000)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)  # 设置最大帧数
    state, info = env.reset(seed=42)
    done = False
    total_reward = 0

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    net1 = torch.nn.Sequential(
        torch.nn.Linear(4, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 2)
    )
    net1.load_state_dict(torch.load("dqn_cartpole.pth", map_location=device))
    net1.eval()

    while not done:
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        action = net1(state_tensor).argmax().item()
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done or truncated:
            break


# 修正后的测试代码
def test_agent(model_path):
    # 创建测试环境，设置渲染模式
    env1 = gym.make('CartPole-v1', render_mode="human", max_episode_steps=1000)

    state, info = env1.reset(seed=42)
    done = False
    truncated = False
    total_reward = 0

    # 确定设备
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 重要：使用与训练时相同的Qnet类来创建网络
    net1 = torch.nn.Sequential(
        torch.nn.Linear(4, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 2)
    ).to(device)
    # 加载模型权重
    net1.load_state_dict(torch.load(model_path, map_location=device))
    # 设置为评估模式
    net1.eval()

    while not done and not truncated:  # 同时检查done和truncated
        # 将状态转换为张量并移动到正确设备
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        # 不计算梯度，提高效率
        with torch.no_grad():
            action = net1(state_tensor).argmax().item()

        state, reward, done, truncated, info = env1.step(action)
        total_reward += reward

    # 输出测试结果
    print(f"测试总奖励: {total_reward}")

    # 关闭环境，释放资源
    env1.close()

    return total_reward


if __name__ == '__main__':
    # main()
    test_agent("dqn_cartpole.pth")



