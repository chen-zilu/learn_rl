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


class DynaQ:
    def __init__(self, env: CliffWalkingEnv, alpha=0.1, gamma=0.9, eps=0.1, n_planning=20):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.n_planning = n_planning

        self.action_num = len(self.env.action_space)
        self.qt = np.zeros((self.env.nrow * self.env.ncol, self.action_num))

        self.mem = {}  # 记录环境模型

    def take_action(self, s):
        if np.random.random() < self.eps:
            return np.random.choice(self.env.action_space)
        else:
            # max_actions = np.flatnonzero(self.qt[s] == np.max(self.qt[s]))
            # return np.random.choice(max_actions)
            return np.argmax(self.qt[s])

    def update(self, s, a, r, sn, an):
        self.qt[s][a] += self.alpha * (r + self.gamma * np.max(self.qt[sn]) - self.qt[s][a])
        self.mem[(s, a)] = (r, sn)  # 记录环境模型

        for _ in range(self.n_planning):
            (ms, ma), (mr, ms_) = random.choice(list(self.mem.items()))
            self.qt[ms][ma] += self.alpha * (mr + self.gamma * np.max(self.qt[ms_]) - self.qt[ms][ma])

    def best_action(self, state):  # 用于打印策略
        q_max = np.max(self.qt[state])
        a = [0 for _ in range(self.action_num)]
        for i in range(self.action_num):  # 若两个动作的价值一样,都会记录下来
            if self.qt[state, i] == q_max:
                a[i] = 1
        return a


def DynaQ_CliffWalking(n_planning):
    ncol = 12
    nrow = 4
    env = CliffWalkingEnv(ncol, nrow)
    np.random.seed(0)
    epsilon = 0.01
    alpha = 0.1
    gamma = 0.9
    agent_q = DynaQ(env, alpha, gamma, epsilon, n_planning)
    num_episodes = 300  # 智能体在环境中运行的序列的数量

    return_list = []  # 记录每一条序列的回报
    for i in range(10):  # 显示10个进度条
        # tqdm的进度条功能
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
                episode_return = 0
                s = env.reset()
                done = False

                # q learning
                while not done:
                    a = agent_q.take_action(s)
                    an=a
                    sn, r, done = env.step(a)
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

    return return_list


def main():
    np.random.seed(0)
    random.seed(0)
    n_planning_list = [0, 2, 20]
    for n_planning in n_planning_list:
        print('Q-planning步数为：%d' % n_planning)
        import time
        time.sleep(0.5)
        return_list = DynaQ_CliffWalking(n_planning)
        episodes_list = list(range(len(return_list)))
        plt.plot(episodes_list,
                 return_list,
                 label=str(n_planning) + ' planning steps')

    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Dyna-Q on {}'.format('Cliff Walking'))
    plt.show()

if __name__ == '__main__':
    main()
