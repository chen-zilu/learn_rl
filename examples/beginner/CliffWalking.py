import copy

import numpy as np


class CliffWalkingEnv:
    """ 悬崖漫步环境"""
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网格世界的行
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()

    def createP(self):
        # 初始化
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0,
                                                    True)]
                        continue
                    # 其他位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P

class PolicyIteration:
    """ 策略迭代算法 """
    def __init__(self, env, theta=0.01, gamma=0.5):
        self.gamma = gamma
        self.theta = theta
        self.env = env
        self.Pi = np.array([[0.25, 0.25, 0.25, 0.25] for _ in range(env.nrow * env.ncol)])
        self.Pi_pre = self.Pi.copy()
        self.V = np.zeros(self.env.nrow * self.env.ncol)
        self.V_pre = np.zeros_like(self.V)
        self.Q = np.zeros((self.env.nrow * self.env.ncol, 4))
        self.v_out = None
        self.pi_out = None

    def policy_evaluation(self):
        itr_max = 1000
        itr_cnt = 0
        while itr_cnt <= itr_max:
            itr_cnt += 1
            self.V_pre = self.V.copy()
            for i in range(self.env.ncol * self.env.nrow):
                self.V[i]= 0
                for a in range(4):
                    for p, sn, r, d in self.env.P[i][a]:
                        self.V[i] += self.Pi[i][a]*p*(r+self.gamma*self.V_pre[sn])
            if np.max(np.abs(self.V - self.V_pre)) < self.theta:
                break
        print(f'policy evaluation iter times: {itr_cnt}')
        return

    def policy_improvement(self):
        for i in range(self.env.ncol * self.env.nrow):
            for a in range(4):
                self.Q[i][a] = 0
                for p, sn, r, d in self.env.P[i][a]:
                    self.Q[i][a] += p * (r + self.gamma *  self.V[sn])
            max_q = max(self.Q[i])
            cnt = np.sum(self.Q[i] == max_q)
            self.Pi[i] = [1.0/cnt if q == max_q else 0 for q in self.Q[i]]
        return

    def policy_iteration(self):
        itr_max = 1000
        itr_cnt = 0
        while itr_cnt <= itr_max:
            itr_cnt += 1
            self.Pi_pre = self.Pi.copy()
            self.policy_evaluation()
            self.policy_improvement()
            if np.array_equal(self.Pi, self.Pi_pre):
                break
        self.v_out = self.V
        self.pi_out = self.Pi
        print(f'policy iteration iter times: {itr_cnt}')

def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' % ('%.3f' % agent.V[i * agent.env.ncol + j]), end=' ')
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 一些特殊的状态,例如悬崖漫步中的悬崖
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:  # 目标状态
                print('EEEE', end=' ')
            else:
                a = agent.Pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


class ValueIteration:
    """ 策略迭代算法 """
    def __init__(self, env, theta=0.01, gamma=0.5):
        self.gamma = gamma
        self.theta = theta
        self.env = env
        self.Pi = np.array([[0.25, 0.25, 0.25, 0.25] for _ in range(env.nrow * env.ncol)])
        self.Pi_pre = self.Pi.copy()
        self.V = np.zeros(self.env.nrow * self.env.ncol)
        self.V_pre = np.zeros_like(self.V)
        self.Q = np.zeros((self.env.nrow * self.env.ncol, 4))
        self.v_out = None
        self.pi_out = None

    def get_policy(self):
        for i in range(self.env.ncol * self.env.nrow):
            for a in range(4):
                self.Q[i][a] = 0
                for p, sn, r, d in self.env.P[i][a]:
                    self.Q[i][a] += p * (r + self.gamma *  self.V[sn])
            max_q = max(self.Q[i])
            cnt = np.sum(self.Q[i] == max_q)
            self.Pi[i] = [1.0/cnt if q == max_q else 0 for q in self.Q[i]]

        return

    def value_iteration(self):
        itr_max = 1000
        itr_cnt = 0

        while itr_cnt <= itr_max:
            itr_cnt += 1
            self.Pi_pre = self.Pi.copy()
            self.V_pre = self.V.copy()

            for i in range(self.env.ncol * self.env.nrow):
                self.V[i] = 0
                for a in range(4):
                    for p, sn, r, d in self.env.P[i][a]:
                        self.V[i] += self.Pi[i][a] * p * (r + self.gamma * self.V_pre[sn])

            if np.max(np.abs(self.V - self.V_pre)) < self.theta:
                break
        self.get_policy()
        self.v_out = self.V
        self.pi_out = self.Pi
        print(f'value iteration iter times: {itr_cnt}')


def main():
    env = CliffWalkingEnv()
    action_meaning = ['^', 'v', '<', '>']

    # # 策略迭代
    theta = 0.001
    gamma = 0.9
    agent = PolicyIteration(env, theta, gamma)
    agent.policy_iteration()
    print_agent(agent, action_meaning, list(range(37, 47)), [47])

    # 值迭代
    # theta = 0.001
    # gamma = 0.9
    # agent = ValueIteration(env, theta, gamma)
    # agent.value_iteration()
    # print_agent(agent, action_meaning, list(range(37, 47)), [47])


if __name__ == '__main__':
    main()



