# Code based on the gridworld env tutorial https://www.gymlibrary.ml/content/environment_creation/
import gym
from gym import spaces
import numpy as np

class HouseGym(gym.Env):

    def __init__(self):
        """
        The HouseGym has 11 types of symbolic labels:
            00 -> Nothing 
            01 -> Bed 🛏️
            02 -> Computer 🖥️
            03 -> Carpet 🟥
            04 -> Corridor C
            05 -> Fridge 🧊
            06 -> Bedroom B
            07 -> Kitchen K 
            08 -> Toilet T
            09 -> Toilet 🚽
            10 -> Sandwich 🥪
        """
        self.size = 5
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
            }
        )
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        self.map = [[(0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0)],
                    [(0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0)]
                    [(0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0)]
                    [(0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0)]
                    [(0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0)]]
    
    def render():

        pass



if __name__=="__main__":
    env = HouseGym()
    print(env.observation_space.sample())
    print("-------\n|T  🚽|\n|     |\n|   🐲|\n-------")
