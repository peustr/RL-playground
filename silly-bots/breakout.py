import time

import gym
import torch
import torchvision.transforms as tr
import torchvision.transforms.functional as trf

from agents.breakout import BreakoutBot, BreakoutHuman


def state_preprocess(state):
    # Following DeepMind Atari DQN preprocessing.
    transforms = tr.Compose([
        tr.ToPILImage(),
        tr.Grayscale(),
        tr.Resize((110, 84)),
    ])
    return trf.to_tensor(trf.crop(transforms(state), 17, 0, 84, 84))


def bot_extract_frames():
    env_name = 'Breakout-ram-v0'
    # Some good random seeds for our bot, all of them have a score > 200.
    seeds = [14, 15, 56, 134, 146, 172, 197, 201, 282, 342, 389, 439, 467, 482,
             494, 513, 519, 562, 569, 671, 675, 708, 719, 808, 856, 930]
    frame_id = 0
    for seed in seeds:
        env = gym.make(env_name)
        env.seed(seed)
        agent = BreakoutBot()
        total_reward = 0
        state = env.reset()
        pt = state_preprocess(env.unwrapped.ale.getScreenRGB())
        done = False
        while not done:
            action = agent.act(state)
            torch.save(pt, 'data/X/{}.pt'.format(str(frame_id).zfill(6)))
            torch.save(torch.Tensor(action), 'data/y/{}.pt'.format(str(frame_id).zfill(6)))
            state, reward, done, info = env.step(action)
            pt = state_preprocess(env.unwrapped.ale.getScreenRGB())
            total_reward += reward
            frame_id += 1
        if total_reward > 200:
            print(seed, ':', total_reward)


def bot_test_run():
    env_name = 'Breakout-ram-v0'
    num_episodes = 10
    env = gym.make(env_name)
    agent = BreakoutBot()
    total_reward = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            env.render()
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            time.sleep(0.05)
        print('Episode:', episode, 'Total reward:', total_reward)


def human_test_run():
    env_name = 'Breakout-ram-v0'
    num_episodes = 3
    env = gym.make(env_name)
    agent = BreakoutHuman()
    total_reward = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            env.render()
            action = agent.act()
            state, reward, done, info = env.step(action)
            total_reward += reward
            time.sleep(0.05)
        print('Episode:', episode, 'Total reward:', total_reward)


if __name__ == '__main__':
    bot_extract_frames()
    # bot_test_run()
    # human_test_run()
