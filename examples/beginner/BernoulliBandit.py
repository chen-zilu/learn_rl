import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    def __init__(self, K):
        self.K = K
        self.probs = np.random.RandomState(42).rand(K)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]

    def step(self, k):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0
        self.action = []
        self.regrets = []

    def update_regret(self, k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        raise NotImplementedError

    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.action.append(k)
            self.update_regret(k)


class EpsilonGreedy(Solver):
    def __init__(self, bandit, eps, init_prob=1.0):
        super().__init__(bandit)
        self.eps = eps
        self.estimates = np.zeros(self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        self.eps = 1/self.total_count
        if np.random.rand() < self.eps:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1/(self.counts[k] + 1) * (r - self.estimates[k])
        return k

def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


class UCB(Solver):
    """ UCB算法,继承Solver类 """
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1)))  # 计算上置信界
        k = np.argmax(ucb)  # 选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


class ThompsonSampling(Solver):
    """ 汤普森采样算法,继承Solver类 """
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为1的次数
        self._b = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为0的次数

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)  # 按照Beta分布采样一组奖励样本
        k = np.argmax(samples)  # 选出采样奖励最大的拉杆
        r = self.bandit.step(k)

        self._a[k] += r  # 更新Beta分布的第一个参数
        self._b[k] += (1 - r)  # 更新Beta分布的第二个参数
        return k


def eps_main():
    arm_num = 10
    b10arm = BernoulliBandit(arm_num)

    epsilon_greedy_solver = EpsilonGreedy(b10arm, eps=0.1)
    epsilon_greedy_solver.run(5000)
    plt.plot(epsilon_greedy_solver.regrets)
    plt.show()

    return

    # np.random.seed(0)
    epsilons = [0.001, 0.01, 0.1]
    epsilon_greedy_solver_list = [
        EpsilonGreedy(b10arm, eps=e) for e in epsilons
    ]
    epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]

    for solver in epsilon_greedy_solver_list:
        solver.run(5000)

    plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

def ucb_main():
    arm_num = 10
    b10arm = BernoulliBandit(arm_num)
    np.random.seed(1)
    coef = 1  # 控制不确定性比重的系数
    UCB_solver = UCB(b10arm, coef)
    UCB_solver.run(5000)
    plt.plot(UCB_solver.regrets)
    plt.show()


def tms_main():
    arm_num = 10
    b10arm = BernoulliBandit(arm_num)
    np.random.seed(1)
    thompson_sampling_solver = ThompsonSampling(b10arm)
    thompson_sampling_solver.run(5000)
    print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret)
    plot_results([thompson_sampling_solver], ["ThompsonSampling"])

def main():
    # ucb_main()
    # eps_main()
    tms_main()


if __name__ == '__main__':
    main()