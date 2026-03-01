import gymnasium as gym
import numpy as np
import pygame
import sys


class Agent:
    def __init__(self, env):
        self.action_space = env.action_space
        self.state_space = env.observation_space
        self.state_value = np.zeros(env.observation_space.n)
        self.action_value = np.zeros((self.state_space.n, self.action_space.n))
        self.experiences = list()
        self.gamma = 0.98
        self.epsilon = 0.2
        self.method = 'none'

    def take_action_greedy(self, state):
        q_values = self.action_value[state]

        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space.n)
        else:
            max_value = np.max(q_values)
            best_actions = np.flatnonzero(q_values == max_value)
            return np.random.choice(best_actions)

    def take_action(self, state):
        max_value = np.max(self.action_value[state])
        best_actions = np.flatnonzero(self.action_value[state] == max_value)
        return np.random.choice(best_actions)

    def experience_store(self, experience):
        self.experiences.append(experience)

    def train_offline(self):
        return

    def train_online(self, experience):
        return


class SarsaAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        self.gamma = 0.98
        self.epsilon = 0.2
        self.alpha = 0.05
        self.method = 'sarsa'

    def train_online(self, experience):
        s, a, r, s_nx, a_nx, done = experience
        if done:
            target = r
        else:
            target = r + self.gamma * self.action_value[s_nx, a_nx]
        self.action_value[s, a] += self.alpha * (target - self.action_value[s, a])
        self.state_value[s] = self.action_value[s].max()
        return


class MtAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        self.method = 'monte-carlo'
        self.sa_exp_cnt = np.zeros_like(self.action_value)
        self.gamma = 0.98
        self.epsilon = 0.2

    def train_offline(self):
        # first visit
        g = 0
        action_value_once = np.zeros_like(self.action_value)
        visited_mask = np.zeros_like(self.action_value, dtype=bool)

        for exp in reversed(self.experiences):
            s, a, r, s_nx, *_ = exp
            g = r + self.gamma * g
            action_value_once[s, a] = g
            visited_mask[s, a] = True
        self.sa_exp_cnt[visited_mask] += 1
        self.action_value[visited_mask] += (action_value_once[visited_mask] - self.action_value[visited_mask]) / self.sa_exp_cnt[visited_mask]
        self.state_value = self.action_value.max(axis=1)
        self.experiences.clear()
        return


class QlearningAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        self.gamma = 0.98
        self.epsilon = 0.2
        self.alpha = 0.1
        self.method = 'q-learning'

    def train_online(self, experience):
        s, a, r, s_nx, a_nx, done = experience
        if done:
            target = r
        else:
            target = r + self.gamma * self.action_value[s_nx].max()
        self.action_value[s, a] += self.alpha * (target - self.action_value[s, a])
        self.state_value[s] = self.action_value[s].max()
        return


def test_agent(agent):

    env = gym.make('CliffWalking-v0', render_mode="human", max_episode_steps=1000)
    state, info = env.reset()
    ep_return = 0

    while True:
        # env.render()

        action = agent.take_action(state)

        state_nx, reward, terminated, truncated, info = env.step(action)

        state = state_nx
        ep_return += reward

        if terminated or truncated:
            print(f"Test: return = {ep_return}")
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit()


def main():
    EPISODE_NUM = 1000
    MAX_EPISODE_STEPS = 500

    env = gym.make('CliffWalking-v0', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)

    # agent = MtAgent(env)
    # agent = SarsaAgent(env)
    agent = QlearningAgent(env)

    svl = []

    for ep_idx in range(EPISODE_NUM):
        state, info = env.reset()
        ep_return = 0
        action = agent.take_action_greedy(state)

        for step_idx in range(MAX_EPISODE_STEPS):

            state_nx, reward, terminated, truncated, info = env.step(action)
            action_nx = agent.take_action_greedy(state_nx)
            ep_return += reward

            if agent.method in ['monte-carlo']:
                agent.experience_store((state, action, reward, state_nx, action_nx, terminated))
            elif agent.method in ['sarsa', 'q-learning']:
                agent.train_online((state, action, reward, state_nx, action_nx, terminated))

            state = state_nx
            action = action_nx

            if terminated or truncated:
                break

        if agent.method in ['monte-carlo']:
            agent.train_offline()

        svl.append(ep_return)
        print(f"Episode {ep_idx + 1}: Episode return = {ep_return}")

        # svl.append(agent.state_value.sum())
        # print(f"Episode {ep_idx + 1}: State value sum = {agent.state_value.sum()}")

    import matplotlib.pyplot as plt
    plt.plot(svl)
    plt.show()
    # test
    test_agent(agent)

    env.close()


if __name__ == '__main__':
    main()

