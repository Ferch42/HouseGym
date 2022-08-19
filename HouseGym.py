# Code based on the gridworld env tutorial https://www.gymlibrary.ml/content/environment_creation/
import gym
from gym import spaces
import numpy as np
import os
import time
from LTL import *

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
            08 -> Bathroom T
            09 -> Toilet 🚽
            10 -> Sandwich 🥪
        """
        self.size = 5
        self.observation_space = spaces.Box(0, self.size - 1, shape=(2,), dtype=int)
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),  # DOWN
            1: np.array([0, 1]),  # RIGHT
            2: np.array([-1, 0]), # UP
            3: np.array([0, -1]), # LEFT
        }
        
        self.emoji = {
            0: '  ', 1: '🛏️ ', 2: '💻', 3: '🟥', 4: 'C ', 5: '🧊', 6: 'B ', 7: 'K ', 8: 'T ', 9: '🚽', 10: '🥪', 'agent': '🐲'
        }
        self.symbol = {
            0: "Nothing", 1: "Bed", 2: "Computer", 3: "Carpet", 4: "Corridor", 5: "Fridge", 6: "Bedroom", 7: "Kitchen", 8: "Bathroom", 9: "Toilet", 10: "Sandwich"
        }
        # Map is a matrix of tuples ("LOCATION", "FLOORTYPE", "OBJECT")
        self.map = [[(8,0,9), (4,3,0), (6,0,0), (6,0,1), (6,0,10)],
                    [(8,0,0), (4,3,0), (6,0,0), (6,0,0), (6,0,2)],
                    [(8,0,0), (4,3,0), (6,0,0), (6,0,0), (6,0,0)],
                    [(8,0,0), (4,3,0), (6,0,0), (7,0,0), (7,0,0)],
                    [(8,0,0), (4,3,0), (4,3,0), (7,0,0), (7,0,5)]]
        
        self.tasks = [
            ('UNTIL', 'TRUE', "Sandwich"), ('UNTIL', 'TRUE', "Fridge"), ('UNTIL', 'TRUE', "Computer"),
            ('UNTIL', 'TRUE', ('Fridge', 'AND', ('UNTIL', 'TRUE', 'Toilet')))
        ]

        self.reset()

    def get_current_task(self):
        return self.current_task
    
    def render(self):

        os.system('cls||clear')
        for i,row in enumerate(self.map):
            print('-'*(8*self.size- self.size+ 1))
            row_string  = '|'
            agent_string = '|'
            for j, element in enumerate(row):
                element_string = self.emoji[element[0]] + self.emoji[element[1]] + self.emoji[element[2]] + '|'
                row_string+= element_string
                if i == self.__agent_position[0] and j == self.__agent_position[1]:
                    agent_string += f'    {self.emoji["agent"]}|'
                else:
                    agent_string += '      |'
            print(row_string)
            print('|'+ '      |'*5)
            print(agent_string)
        print('-'*(8*self.size- self.size+ 1))

    



    def get_symbols(self):

        return {self.symbol[x] for x in self.map[self.__agent_position[0]][self.__agent_position[1]]}

    def reset(self):

        self.__agent_position = np.array([0,0])
        self.current_task = self.tasks[1]

        observation = (self.__agent_position[0],self.__agent_position[1]) 

        return observation, {}

    
    
    
    def step(self,action):
        
        direction = self._action_to_direction[action]
        self.__agent_position = np.clip(self.__agent_position + direction, 0, self.size -1)
        symbols = self.get_symbols()
        next_task = prog(symbols,self.current_task)
        
        observation = (self.__agent_position[0],self.__agent_position[1]) 
        reward = 0
        done = False
        
        if type(next_task) == bool:
            # Task complete
            done = True
            if next_task:
                reward = 1
            else:
                reward = -1

        self.current_task = next_task
        
        return observation, reward, done, {}





if __name__=="__main__":
    env = HouseGym()
    env.reset()
    print(env.observation_space.sample())

    while True:
        x = env.step(np.random.randint(0,4))
        env.render()
        print(env.get_symbols())
        print(x)
        done = x[2]
        if done:
            break
        time.sleep(0.1)
