import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from examples import rl_utils


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 device):
        self.policy_net = PolicyNet(state_dim, hidden_dim,
                                    action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.device = device

    def take_action(self, state):  # 根据动作概率分布随机采样
        state = torch.as_tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor([state_list[i]],
                                 dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降


# 修正后的测试代码
def test_agent(net_type, model_path):
    # 创建测试环境，设置渲染模式
    env1 = gym.make('CartPole-v1', render_mode="human")
    # 设置最大步数限制
    env1 = gym.wrappers.TimeLimit(env1, max_episode_steps=1000)

    state, info = env1.reset(seed=42)
    done = False
    truncated = False
    total_reward = 0

    # 确定设备
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 重要：使用与训练时相同的Qnet类来创建网络
    net1 = torch.load('reinforce_cartpole_full.pth', weights_only=False).to(device)

    # net1 = PolicyNet(state_dim=4, hidden_dim=128, action_dim=2).to(device)
    # # 加载模型权重
    # net1.load_state_dict(torch.load(model_path, map_location=device))
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


def main():
    learning_rate = 1e-3
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env_name = "CartPole-v1"
    env = gym.make(env_name)
    env.reset(seed=0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma,
                      device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                state, info = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, truncted, _ = env.step(action)
                    done = done or truncted
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    torch.save(agent.policy_net, 'reinforce_cartpole_full.pth')

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on {}'.format(env_name))
    plt.show()


if __name__ == '__main__':
    # main()
    test_agent(PolicyNet, 'reinforce_carpole_full.pth')
