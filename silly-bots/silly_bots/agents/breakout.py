import keyboard

import numpy as np
import torch
from torch.optim import Adam

from silly_bots.models import B8MLP


class BreakoutHuman(object):
    def act(self):
        if keyboard.is_pressed('space'):
            return 1  # FIRE
        if keyboard.is_pressed('right arrow'):
            return 2
        if keyboard.is_pressed('left arrow'):
            return 3
        return 0  # NOP


class BreakoutBot(object):
    """ A very simple bot that has decent performance on breakout that
        uses the RAM for state representation. Just reads the bytes for
        the plank's and ball's x position and tries to align them.
    """
    def __init__(self, plank_offset=8, plank_speed=6):
        self.plank_x_byte = 72
        self.ball_x_byte = 99
        self.plank_offset = plank_offset
        self.plank_speed = plank_speed
        self.ball_x = -1

    def act(self, state):
        if self.ball_x == int(state[self.ball_x_byte]):
            return 1
        self.ball_x = int(state[self.ball_x_byte])
        self.plank_x = int(state[self.plank_x_byte]) + self.plank_offset
        if self.plank_x - self.ball_x < -self.plank_speed:
            return 2
        if self.plank_x - self.ball_x > self.plank_speed:
            return 3
        return 0


class BreakoutQBot(object):
    def __init__(self, train=True, epsilon=0.95, gamma=0.99):
        self.model = B8MLP(4)
        self.optimizer = Adam(self.model.parameters())
        self.memory = []
        self.epsilon = epsilon
        self.gamma = gamma

    def act(self, state):
        if not self.train:  # No training = no exploration.
            return self.model(state).argmax().item()
        if np.random.rand() > self.epsilon:
            return np.random.choice([0, 1, 2, 3])
        return self.model(state).argmax().item()

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def reset_memory(self):
        self.memory = []

    def train(self):
        for state, action, reward, next_state in self.memory:
            predicted_reward = self.model(state)[action]
            done = next_state is None
            if done:
                expected_reward = reward
            else:
                expected_reward = reward + self.gamma * self.model(next_state).max()
            self.optimizer.zero_grad()
            loss = torch.square(predicted_reward - expected_reward)
            loss.backward()
            self.optimizer.step()
