# Code based on the gridworld env tutorial https://www.gymlibrary.ml/content/environment_creation/
import gym
from gym import spaces
import numpy as np

class HouseGym(gym.Env):

    def __init__(self):
        """
        The HouseGym has 10 types of symbolic labels:
            00 -> Nothing 
            01 -> Bed 🛌🏻
            02 -> Computer 💻
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

        self.emoji = {
            0: '  ', 1: '🛏️ ', 2: '💻', 3: '🟥', 4: 'C ', 5: '🧊', 6: 'B ', 7: 'K ', 8: 'T ', 9: '🚽', 10: '🥪', 'agent': '🐲'
        }
        # Map is a matrix of tuples ("LOCATION", "FLOORTYPE", "OBJECT")
        self.map = [[(8,0,9), (4,3,0), (6,0,0), (6,0,1), (6,0,10)],
                    [(8,0,0), (4,3,0), (6,0,0), (6,0,0), (6,0,2)],
                    [(8,0,0), (4,3,0), (6,0,0), (6,0,0), (6,0,0)],
                    [(8,0,0), (4,3,0), (6,0,0), (7,0,0), (7,0,0)],
                    [(8,0,0), (4,3,0), (4,3,0), (7,0,0), (7,0,5)]]
    
    def render(self):

        for row in self.map:
            print('-'*(8*self.size- self.size+ 1))
            row_string  = '|'
            for element in row:
                element_string = self.emoji[element[0]] + self.emoji[element[1]] + self.emoji[element[2]] + '|'
                row_string+= element_string
            print(row_string)
            print('|'+ '      |'*5)
            print('|'+ '      |'*5)
        print('-'*(8*self.size- self.size+ 1))




if __name__=="__main__":
    env = HouseGym()
    print(env.observation_space.sample())

    env.render()
