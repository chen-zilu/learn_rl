import matplotlib.pyplot as plt
from matplotlib import animation
import torch

agent.q_net.load_state_dict(torch.load("../DQN/dqn_cartpole.pth", map_location=device))
agent.q_net.eval()

frames = []

state, info = env.reset()
done = False
while not done:
    frames.append(env.render())  # 保存当前帧
    action = agent.take_action(state, epsilon=0)  # 评估时不探索
    state, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break

env.close()

# --- 动画部分 ---
fig = plt.figure()
plt.axis('off')
img = plt.imshow(frames[0])

def update(frame):
    img.set_data(frame)
    return [img]

ani = animation.FuncAnimation(fig, update, frames=frames, interval=40, blit=True)
plt.show()
