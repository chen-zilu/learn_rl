import gymnasium as gym
import numpy as np

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


if __name__ == '__main__':
    env = gym.make("FrozenLake-v1", render_mode="ansi", is_slippery=True)

    state, info = env.reset()  # 必须先reset才能有初始状态
    env = env.unwrapped  # 解封装才能访问状态转移矩阵P
    print(env.render())  # 环境渲染,通常是弹窗显示或打印出可视化的环境

    holes = set()
    ends = set()
    for s in env.P:
        for a in env.P[s]:
            for s_ in env.P[s][a]:
                if s_[2] == 1.0:  # 获得奖励为1,代表是目标
                    ends.add(s_[1])
                if s_[3] == True:
                    holes.add(s_[1])
    holes = holes - ends
    print("冰洞的索引:", holes)
    print("目标的索引:", ends)

    for a in env.P[14]:  # 查看目标左边一格的状态转移信息
        print(env.P[14][a])

    action_meaning = ['<', 'v', '>', '^']

    # 策略迭代
    # theta = 1e-5
    # gamma = 0.9
    # agent = PolicyIteration(env, theta, gamma)
    # agent.policy_iteration()
    # print_agent(agent, action_meaning, [5, 7, 11, 12], [15])

    # 值迭代
    theta = 1e-5
    gamma = 0.9
    agent = ValueIteration(env, theta, gamma)
    agent.value_iteration()
    print_agent(agent, action_meaning, [5, 7, 11, 12], [15])

