# Code based on the gridworld env tutorial https://www.gymlibrary.ml/content/environment_creation/
import gym
from gym import spaces
import numpy as np
import os
import time
from LTL import *


LIGHT_INDEX = 3
MUSIC_INDEX = 4
MONKEY_INDEX = 1

class HouseGym_2(gym.Env):

    def __init__(self):
        """
        The HouseGym has 10 types of symbolic labels:
            00 -> Nothing 
            01 -> Monkey 🐒
            02 -> Mokey_Scared 🙈
            03 -> Light 💡
            04 -> Music 🎵
            05 -> LightSwitch 🔦
            06 -> Radio 📻
            07 -> Ball ⚽
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
            0: '  ', 1: '🐒', 2: '🙈', 3: '💡', 4: '🎵', 5: '🔦', 6: '📻', 7: '⚽', 8: 'T ', 9: '🚽', 10: '🥪', 'agent': '🐲'
        }
        self.symbol = {
            0: "Nothing", 1: "Monkey", 2: "Mokey_Scared", 3: "Light", 4: "Music", 5: "LightSwitch", 6: "Radio", 7: "Ball", 8: "Bathroom", 9: "Toilet", 10: "Sandwich"
        }
        # Map is a matrix of tuples ("LOCATION", "FLOORTYPE", "OBJECT")
        self.map = [[(0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0)],
                    [(0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0)],
                    [(0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,5)],
                    [(0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0)],
                    [(0,0,6), (0,0,0), (0,0,0), (0,0,0), (0,0,7)]]
        
        self.tasks = [
            ('UNTIL', 'TRUE', ('AND', 'LightSwitch',('NEXT',('UNTIL', 'Light', 'Radio')))),
            ('UNTIL', 'TRUE', ('AND', 'LightSwitch',('NEXT',('UNTIL', 'Light', ('AND', 'Radio', ('NEXT',('UNTIL', 'Music', 'LightSwitch'))))))),
            ('UNTIL', 'TRUE', ('AND', 'LightSwitch',('NEXT',('UNTIL', 'Light', ('AND', 'Radio', ('NEXT',('UNTIL', 'Music', ('AND', 'LightSwitch', ('NEXT', ('UNTIL', ('AND', 'Music', ('NOT', 'Light')), 'Ball')))))))))),
            ('UNTIL', 'TRUE', ('AND', 'LightSwitch',('NEXT',('UNTIL', 'Light', ('AND', 'Radio', ('NEXT',('UNTIL', 'Music', ('AND', 'LightSwitch', ('NEXT', ('UNTIL', ('AND', 'Music', ('NOT', 'Light')), ('AND', 'Ball', ('NEXT', 'Monkey')))))))))))),
            ('UNTIL', ('AND', ('NOT', 'Light'), ('AND', ('NOT', 'Music'), ('NOT', 'Monkey'))), ('AND', 'LightSwitch', ('NEXT', ('UNTIL', ('AND', 'Light', ('AND', ('NOT', 'Music'), ('NOT', 'Monkey'))), ('AND', 'Radio', ('NEXT', ('UNTIL', ('AND', 'Light', ('AND', 'Music', ('NOT', 'Monkey'))), ('AND', 'LightSwitch', ('NEXT', ('UNTIL', ('AND', ('NOT', 'Light'), ('AND', 'Music', ('NOT', 'Monkey'))), ('AND', 'Ball', ('NEXT', ('AND', ('NOT', 'Light'), ('AND', 'Music', 'Monkey'))))))))))))))
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
                if i==0 and j==0:
                    element_string = self.emoji[LIGHT_INDEX* self.LIGHT] + self.emoji[MUSIC_INDEX* self.MUSIC] + self.emoji[MONKEY_INDEX* self.MONKEY] + '|'
                else:
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

        symbols_set = {self.symbol[x] for x in self.map[self.__agent_position[0]][self.__agent_position[1]]}
        if self.LIGHT:
            symbols_set.add('Light')
        if self.MUSIC:
            symbols_set.add('Music')
        if self.MONKEY:
            symbols_set.add('Monkey')

        return symbols_set 

    def reset(self):

        self.__agent_position = np.array([0,0])
        self.current_task = self.tasks[4]
        self.LIGHT = False
        self.MUSIC = False
        self.MONKEY = False

        observation = (self.__agent_position[0],self.__agent_position[1]) 

        return observation, {}

    
    
    
    def step(self,action):
        
        direction = self._action_to_direction[action]
        self.__agent_position = np.clip(self.__agent_position + direction, 0, self.size -1)
        symbols = self.get_symbols()
        
        if 'LightSwitch' in symbols:
            self.LIGHT = not self.LIGHT
        if self.LIGHT and 'Radio' in symbols:
            self.MUSIC = not self.MUSIC
        if not self.LIGHT and self.MUSIC and 'Ball' in symbols:
            self.MONKEY = not self.MONKEY

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
    env = HouseGym_2()
    env.reset()
    print(env.observation_space.sample())

    while True:
        x = env.step(int(input('?')))
        env.render()
        print(env.get_symbols())
        print(env.get_current_task())
        print(x)
        done = x[2]
        if done:
            break
        time.sleep(1)
