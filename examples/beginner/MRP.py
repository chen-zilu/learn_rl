import numpy as np


# 给定一条序列,计算从某个索引（起始状态）开始到序列最后（终止状态）得到的回报
def compute_return(start_index, chain, rewards, gamma):
    G = 0
    for i in reversed(range(start_index, len(chain))):
        G = gamma * G + rewards[chain[i] - 1]
    return G


def compute_state_value(P, rewards, gamma):
    # V = R + gamma * P * V
    I = np.eye(len(rewards))
    R = np.array(rewards).reshape(-1, 1)
    V = np.linalg.inv(I-gamma*P).dot(R)
    return V


def main():
    np.random.seed(0)
    # 定义状态转移概率矩阵P
    P = [
        [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
        [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
        [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    P = np.array(P)

    rewards = [-1, -2, -2, 10, 1, 0]  # 定义奖励函数
    gamma = 0.5  # 定义折扣因子

    # 一个状态序列,s1-s2-s3-s6
    chain = [1, 2, 3, 6]
    start_index = 0
    G = compute_return(start_index, chain, rewards, gamma)
    V = compute_state_value(P, rewards, gamma)
    print("根据本序列计算得到回报为：%s。" % G)
    print("根据状态转移概率矩阵计算得到各状态的状态值为：\n%s" % V)

    res = V[0] - rewards[0] - gamma * np.array(P[0, :]).dot(V)
    print(f'{res=}')
    print(f'{V[0]=}')



if __name__ == '__main__':
    main()
