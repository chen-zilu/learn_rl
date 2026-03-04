# RL学习过程

## 搭建环境

- 状态离散-动作离散：CliffWalking-v0，方法包括 Monte-Carlo、Sarsa、Q-Learning、REINFORCE.
- 状态连续-动作离散：CartPole-v1，方法包括 DQN.
- 状态连续-动作连续：LunarLanderContinuous-v3，方法包括 A2C、DDPG、PPO、SAC.

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

状态空间连续，利用神经网络拟合动作值$q(s,a,\omega)$

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

神经网络直接输出策略$\pi(s,a,\theta)$

常用的目标函数有两类：

- 平均状态值

  $$
  J(\theta)=\bar{v}_{\pi} \triangleq \sum_s d(s)v_{\pi}(s)
  $$

  按照是否与策略相关，$d$ 分为 $d_0$ 和 $d_\pi$，可以是均匀分布、侧重起始状态、稳态分布

  第二种形式：

  $$
  \mathbb{E}\left[ \sum_{t=0} ^\infty \gamma^t R_{t+1} \right]
  $$
- 单步奖励：

  $$
  J(\theta)=\bar{r}_{\pi} \triangleq \sum_s d_{\pi}(s)r_{\pi}(s) = \sum_s \left[ d_{\pi}(s) \sum_a \left( \pi(a|s)  \sum_r r(s,a)p(r|s,a) \right) \right]
  $$

  第二种形式（长时间采样后和初始状态无关）：

  $$
  \lim_{n \to \infty } \frac{1}{n} \mathbb{E} \left[ \sum_{k=1} R_{t+k} | S=s_0 \right]
  =\lim_{n \to \infty } \frac{1}{n} \mathbb{E} \left[ \sum_{k=1} R_{t+k} \right]
  =\bar{r}_\pi
  $$
- 两类目标函数的关系：

$$
\bar{r}_\pi = (1-\gamma)\bar{v}_\pi
$$

梯度计算引入对数转换为期望， 并用softmax函数限制在 ln 函数定义域内。

$$
\begin{aligned}
\nabla J(\theta) &\to \sum_s \eta(s) \sum_a \nabla_\theta \pi(a|s,\theta) q_\pi(s,a) \\
&=\mathbb{E} \left( \nabla_\theta \ln(\pi(A|S,\theta)q_\pi(S,A) \right)
\end{aligned}
$$

这类undeterministic方法用神经网络输入状态 s，输出每个动作的概率，并用softmax归一化，动作空间仍是离散的。deterministic方法直接输出动作，动作空间可以是连续的。

### REINFORCE算法

on-policy，估计动作值，对policy net进行梯度上升优化更新

1. 采用蒙特卡洛法更新动作值
   $$
   q(s_t,a_t)=\sum_{k=t+1}^T \gamma^{k-t-1} r_k
   $$
2. 对policy net参数进行更新
   $$
   \theta_{t+1} = \theta_t + \alpha \nabla_\theta \ln\pi(a_t|s_t,\theta_t)q_t(s_t,a_t)
   $$

注意：

- G值可以取平均值做 **baseline**
- 采样的策略和训练的策略需要是同一个策略，可以一个策略采样多个episode用于 batch 训练
- loss 用 $-\ln\pi q$ 计算
- 策略更新后记得把**经验清空**
- 用 **every_visit** 采样数据更多
- 从动作概率采样用：torch.multinomial

### PPO算法

### A2C算法

###SAC算法
