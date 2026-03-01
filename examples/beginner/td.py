import random

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # tqdm是显示循环进度条的库


class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0  # 记录当前智能体位置的横坐标
        self.y = self.nrow - 1  # 记录当前智能体位置的纵坐标
        self.action_space = [0, 1, 2, 3]  # 四个动作

    def step(self, action):  # 外部调用这个函数来改变当前位置
        # 4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:  # 下一个位置在悬崖或者目标
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    def reset(self):  # 回归初始状态,坐标轴原点在左上角
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x


class Sarsa:
    def __init__(self, env: CliffWalkingEnv, alpha=0.1, gamma=0.9, eps=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.action_num = len(self.env.action_space)
        self.qt = np.zeros((self.env.nrow * self.env.ncol, self.action_num))

    def take_action(self, s):
        # 选取策略
        if np.random.random() < self.eps:
            a = np.random.choice(self.env.action_space)
        else:
            a = np.argmax(self.qt[s])
        return a

    def update(self, s, a, r, sn, an):
        self.qt[s][a] += self.alpha * (r + self.gamma * self.qt[sn][an] - self.qt[s][a])

    def best_action(self, state):  # 用于打印策略
        q_max = np.max(self.qt[state])
        a = [0 for _ in range(self.action_num)]
        for i in range(self.action_num):  # 若两个动作的价值一样,都会记录下来
            if self.qt[state, i] == q_max:
                a[i] = 1
        return a


class SarsaNstep:
    def __init__(self, env: CliffWalkingEnv, alpha=0.1, gamma=0.9, eps=0.1, step_num=5):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        self.step_num = step_num
        self.step_reward = []  # 记录n步的奖励
        self.step_sa = []  # 记录n步的(s,a)

        self.action_num = len(self.env.action_space)
        self.qt = np.zeros((self.env.nrow * self.env.ncol, self.action_num))

    def take_action(self, s):
        # 选取策略
        if np.random.random() < self.eps:
            a = np.random.choice(self.env.action_space)
        else:
            a = np.argmax(self.qt[s])
        return a

    def update(self, s, a, r, sn, an, done):
        self.step_reward.append(r)
        self.step_sa.append((s, a))

        if len(self.step_reward) > self.step_num:
            reward_sum = self.qt[sn][an]
            for i in reversed(range(len(self.step_reward))):
                reward_sum = reward_sum * self.gamma + self.step_reward[i]
                # 如果到达终止状态,最后几步虽然长度不够n步,也将其进行更新
                if done and i > 0:
                    s_tmp, a_tmp = self.step_sa[i]
                    self.qt[s_tmp, a_tmp] += self.alpha * (reward_sum - self.qt[s_tmp, a_tmp])

            s_ns, a_ns = self.step_sa.pop(0)
            self.qt[s_ns][a_ns] += self.alpha * (reward_sum - self.qt[s_ns][a_ns])
            self.step_reward.pop(0)

        if done:
            self.step_reward.clear()
            self.step_sa.clear()

    def best_action(self, state):  # 用于打印策略
        q_max = np.max(self.qt[state])
        a = [0 for _ in range(self.action_num)]
        for i in range(self.action_num):  # 若两个动作的价值一样,都会记录下来
            if self.qt[state, i] == q_max:
                a[i] = 1
        return a


def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


class QLearning:
    def __init__(self, env: CliffWalkingEnv, alpha=0.1, gamma=0.9, eps=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        self.action_num = len(self.env.action_space)
        self.qt = np.zeros((self.env.nrow * self.env.ncol, self.action_num))

    def take_action(self, s):
        if np.random.random() < self.eps:
            return np.random.choice(self.env.action_space)
        else:
            max_actions = np.flatnonzero(self.qt[s] == np.max(self.qt[s]))
            return np.random.choice(max_actions)

    def update(self, s, a, r, sn, an):
        self.qt[s][a] += self.alpha * (r + self.gamma * np.max(self.qt[sn]) - self.qt[s][a])

    def best_action(self, state):  # 用于打印策略
        q_max = np.max(self.qt[state])
        a = [0 for _ in range(self.action_num)]
        for i in range(self.action_num):  # 若两个动作的价值一样,都会记录下来
            if self.qt[state, i] == q_max:
                a[i] = 1
        return a


def main():
    ncol = 12
    nrow = 4
    env = CliffWalkingEnv(ncol, nrow)
    np.random.seed(0)
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    agent = Sarsa(env, alpha, gamma, epsilon)
    agent_nstep = SarsaNstep(env, alpha, gamma, epsilon)
    agent_q = QLearning(env, alpha, gamma, epsilon)
    num_episodes = 500  # 智能体在环境中运行的序列的数量

    return_list = []  # 记录每一条序列的回报
    for i in range(10):  # 显示10个进度条
        # tqdm的进度条功能
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
                episode_return = 0
                s = env.reset()
                done = False

                # sarsa
                # a = agent.take_action(s)
                # while not done:
                #     sn, r, done = env.step(a)
                #     an = agent.take_action(sn)
                #     episode_return += r  # 这里回报的计算不进行折扣因子衰减
                #     agent.update(s, a, r, sn, an)
                #     s = sn
                #     a = an

                # sarsa nstep
                # a = agent_nstep.take_action(s)
                # while not done:
                #     sn, r, done = env.step(a)
                #     an = agent_nstep.take_action(sn)
                #     episode_return += r  # 这里回报的计算不进行折扣因子衰减
                #     agent_nstep.update(s, a, r, sn, an, done)
                #     s = sn
                #     a = an

                # q learning
                a = agent_q.take_action(s)
                while not done:
                    sn, r, done = env.step(a)
                    an = agent_q.take_action(sn)
                    episode_return += r  # 这里回报的计算不进行折扣因子衰减
                    agent_q.update(s, a, r, sn, an)
                    s = sn
                    a = an

                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    action_meaning = ['^', 'v', '<', '>']
    print('Sarsa算法最终收敛得到的策略为：')
    print_agent(agent_q, env, action_meaning, list(range(37, 47)), [47])

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Sarsa on {}'.format('Cliff Walking'))
    plt.show()


if __name__ == '__main__':
    main()
