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



S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]  # 动作集合
# 状态转移函数
P = {
    "s1-保持s1-s1": 1.0,
    "s1-前往s2-s2": 1.0,
    "s2-前往s1-s1": 1.0,
    "s2-前往s3-s3": 1.0,
    "s3-前往s4-s4": 1.0,
    "s3-前往s5-s5": 1.0,
    "s4-前往s5-s5": 1.0,
    "s4-概率前往-s2": 0.2,
    "s4-概率前往-s3": 0.4,
    "s4-概率前往-s4": 0.4,
}
# 奖励函数
R = {
    "s1-保持s1": -1,
    "s1-前往s2": 0,
    "s2-前往s1": -1,
    "s2-前往s3": -2,
    "s3-前往s4": -2,
    "s3-前往s5": 0,
    "s4-前往s5": 10,
    "s4-概率前往": 1,
}
gamma = 0.5  # 折扣因子
MDP = (S, A, P, R, gamma)

# 策略1,随机策略
Pi_1 = {
    "s1-保持s1": 0.5,
    "s1-前往s2": 0.5,
    "s2-前往s1": 0.5,
    "s2-前往s3": 0.5,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.5,
    "s4-概率前往": 0.5,
}
# 策略2
Pi_2 = {
    "s1-保持s1": 0.6,
    "s1-前往s2": 0.4,
    "s2-前往s1": 0.3,
    "s2-前往s3": 0.7,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.1,
    "s4-概率前往": 0.9,
}


# 把输入的两个字符串通过“-”连接,便于使用上述定义的P、R变量
def join(str1, str2):
    return str1 + '-' + str2

def sample(MDP, Pi, timestep_max, number):
    ''' 采样函数,策略Pi,限制最长时间步timestep_max,总共采样序列数number '''
    S, A, P, R, gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)]  # 随机选择一个除s5以外的状态s作为起点
        # 当前状态为终止状态或者时间步太长时,一次采样结束
        while s != "s5" and timestep <= timestep_max:
            timestep += 1
            rand, temp = np.random.rand(), 0
            # 在状态s下根据策略选择动作
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0)
                if temp > rand:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break
            rand, temp = np.random.rand(), 0
            # 根据状态转移概率得到下一个状态s_next
            for s_opt in S:
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s, a, r, s_next))  # 把（s,a,r,s_next）元组放入序列中
            s = s_next  # s_next变成当前状态,开始接下来的循环
        episodes.append(episode)
    return episodes


# 计算状态值
s1_action_reward = np.array([-1, 0])
s1_action_prob = np.array([0.5, 0.5])
s1_action_transition = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.0],  # 保持s1
    [0.0, 1.0, 0.0, 0.0, 0.0],  # 前往s2
])
s1_transition = s1_action_prob.dot(s1_action_transition)
s1_state_reward = s1_action_prob.dot(s1_action_reward)
print(f'{s1_transition=}')
print(f'{s1_state_reward=}')

P = np.array([
    [0.5, 0.5, 0, 0, 0],
    [0.5, 0, 0.5, 0, 0],
    [0, 0, 0, 0.5, 0.5],
    [0, 0.1, 0.2, 0.2, 0.5],
    [0, 0, 0, 0, 1],
])

R = np.array([
    -0.5,
    -1.5,
    -1,
    5.5,
    0,
])

gamma = 0.5

V = compute_state_value(P, R, gamma)
print(f'{V=}')

# 采样5次,每个序列最长不超过20步
#  [('s2', '前往s1', -1, 's1'), ('s1', '保持s1', -1, 's1'), ...]
episodes = sample(MDP, Pi_1, 20, 10000)

R_est = {s:0 for s in S}
R_est_cnt = {s:0 for s in S}

V = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
N = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}

for ep in episodes:
    G = 0
    # for i in np.flip(range(len(ep)-1)):
    for i in range(len(ep) - 1, -1, -1):  #一个序列从后往前计算
        (s, a, r, s_n) = ep[i]
        G = G*gamma + r
        R_est_cnt[s] += 1
        R_est[s] += (G - R_est[s]) / R_est_cnt[s]

    # for i in range(len(ep) - 1, -1, -1):  # 一个序列从后往前计算
    #     (s, a, r, s_next) = ep[i]
    #     G = r + gamma * G
    #     N[s] = N[s] + 1
    #     V[s] = V[s] + (G - V[s]) / N[s]


print(f'{R_est=}')
print(f'{V=}')



