import keyboard


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
