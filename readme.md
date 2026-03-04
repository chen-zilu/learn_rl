# RL学习过程

## 搭建环境

- 离散环境：CliffWalking-v0
- 连续环境：Pendulum-v1
- 离散环境：CartPole-v1
- 连续环境：MountainCarContinuous-v0

## 基本概念

### 模型和数据

模型本质是状态转移和单步reward，没有模型时采样数据就是为了获取$(s,a) \to (s',r)$对应的概率

### discounted return

$$
G_t=R_{t+1}+\gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

### state value

Definition and Bellman equation:

$$
\begin{aligned}
v_{\pi}(s) &=\mathbb{E}[G_t|S_t = s] \\
&= \mathbb{E}[R_{t+1}|S_t=s]+\gamma \mathbb{E}[G_{t+1}|S_t=s] \\
&= \sum_a \pi(a|s)\big[\sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s, a)v_{\pi}(s')\big]
\end{aligned}
$$

Rewrite as:

$$
v_{\pi}(s) = r_\pi(s) + \gamma \sum_{s'} p_{\pi}(s'|s,a) v_{\pi}(s')
$$

where

$$
r_{\pi}(s) \triangleq \sum_a \pi(a|s) \big[\sum_r p(r|s,a)r \big],~p_{\pi}(s'|s) \triangleq \sum_a \pi(a|s) p(s'|s,a)
$$

Matrix form:

$$
\mathbf{v}_{\pi}=\mathbf{r}_{\pi}+\gamma \mathbf{P}_{\pi} \mathbf{v}_{\pi}
$$

### action value

Definition:

$$
q_{\pi}(s,a)=\mathbb{E}[G_t|S_t=s,A_t=a]
$$

Hence,

$$
v_{\pi}(s)=\sum_a \pi(a|s)q_{\pi}(s,a)
$$

Action value function:

$$
q_{\pi}(s,a)=\sum_r p(r|s, a)r+\sum_{s'}p(s'|s,a)v(s')
$$

### Bellman optimality equation

Definition:

$$
v(s)=\max_{\pi}\sum_a \pi(a|s)q_{\pi}(s,a)
$$

$$


$$

## tabular方法

### 蒙特卡洛方法

从action value的定义出发进行估计。on-policy。

1. policy evalution
   - 采样从(s,a)开始的return（G)估计action value
   - first visit 和 every visit
2. policy improvement
   - 采取动作值最大的策略
   - epsilon greedy

### Sarsa算法

求解bellman方程。on-policy。

1. 采样$(s_t,a_t,r_{t+1},s_{t+1},a_{t+1})$，**$a_{t+1}$是实际的下一动作值**，更新q值，可以进行**多步采样（n-steps）**

$$
q_{t+1}(s_t,a_t)=q_t(s_t,a_t)-\alpha(s_t,a_t)\big[q_t(s_t,a_t)-[r_{t+1} + \gamma q_t(s_{t+1},a_{t+1})] \big]
$$

2. 更新策略: epsilon greedy

### Q-learning

求解bellman最优方程。off-policy，采样的策略可以不一样。

1. 更新q值
   $$
   q_{t+1}(s_t,a_t)=q_{t}(s_t,a_t)-\alpha_t(s_t,a_t)\big[q_t(s,a)-[r_{t+1}+ \gamma \max_a q_t(s_{t+1},a)] \big]
   $$

## value function approximation方法

### DQN算法

利用神经网络拟合动作值。

$$
J=\mathbb{E}\big[ \big( R+\gamma \max_a \hat{q}(s,a,w_t) - \hat{q}(S,A,w) \big)^2 \big] \\
\nabla_w J = \mathbb{E}\big[ \big(R+\gamma \max_a \hat{q}(s,a,w_t) - \hat{q}(S,A,w)\big) \nabla_w \hat{q}(S,A,w)\big]
$$

1. 两个网络：main network+target network
2. 用探索性策略$\pi_b$采样
3. 取小批量样本，对main network进行训练，周期性更新target network

注意：

- 先**热身**，存储一定量experience后开始训练
- 用**纯探索**的策略采集经验训练会很慢，epsilon-greedy更合适
- experience用**deque**, 新增一个数据就取样训练一个batch
- **tensor的转换**有时候要经过 nd.array
- **opimizer有惯性**，不要每次都重置
- 不参与梯度计算的用**detach**
- 推理时：输入最好 **unsqueeze**匹配batch维度；**with no_grad**提高速度；
- **terminated**的reward特殊处理
- 实际常常**收集一定数据**再训练一次而不是每个数据训练一次
- gather算子
- to device

## policy gradient / actor-critic方法

### REINFORCE算法

### PPO算法

### A2C算法

###SAC算法
