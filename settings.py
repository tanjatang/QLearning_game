from enum import Enum
import numpy as np
import os


class Action(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class Color:
    BACKGROUND = (0, 0, 0)
    BORDER = (100, 100, 100)
    END_POSITION_GOOD = (0, 255, 0)
    END_POSITION_BAD = (255, 0, 0)
    END_POSITION_TEXT = (0, 0, 0)
    EMPTY_FIELD = (255, 255, 255)
    WALL = (0, 0, 0)
    AGENT = (0, 0, 255)  # not used currently
    BEST_ARROW_OUTLINE = BORDER
    HOVER_COLOR = (0, 0, 255)
    INFO_TEXT = (255, 255, 255)
    TRANSPARENT = (0, 0, 0, 0)

    @staticmethod
    def for_q_value(q_value: float, max_q_value: float):

        # No arrow will be drawn if None is returned
        if q_value == 0:
            return None

        # tanh mapping squeezes the values between 0 and 1.
        # A multiplication factor is calculated to stretch the Q-values with respect to
        # the highest absolute Q-value. This improves visualization
        color_value = int(abs(np.tanh(q_value * (np.arctanh(0.9999) / max_q_value))) * 255)

        assert 0 <= color_value <= 255

        if q_value > 0:
            # green channel
            color = (0, 255, 0, color_value)
        else:
            # red channel
            color = (255, 0, 0, color_value)
        return color


class FieldType:
    EMPTY = 0
    END_POS = 1
    BLOCKED = 2


class SaveSettings:
    ENABLED = True
    INTERVAL = 100
    SAVE_PATH = "checkpoints"


class RenderSettings:
    TIME_BETWEEN_FRAMES = 300  # ms
    ENABLED = True
    INTERVAL = 50  # epochs
    GAME_PADDING = 30  # pixels
    PIXEL_PER_FIELD = 100  # pixels
    FONT_NAME = "assets/fonts/arial.ttf"
    FONT_SIZE = 30  # pt
    INFO_FONT_SIZE = 20
    ARROW_WIDTH = 0.2  # percentage of field pixels
    AGENT_SIZE = int(0.8 * PIXEL_PER_FIELD)  # pixels
    AGENT_IMAGE_PATH_GOOD = os.path.join("assets", "robot.png")
    AGENT_IMAGE_PATH_BAD = os.path.join("assets", "robot_broken.png")
    ARROW_OUTLINE_WIDTH = int(PIXEL_PER_FIELD / 20)  # pixels
    UPDATE_FREQ_TITLE = 10  # epochs
    INFO_HEIGHT = 100



