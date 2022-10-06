from HouseGym import HouseGym
import numpy as np
import time
import random
from LTL import *


N_EPISODES = 100
N_TIMESTEPS = 30
EPSILON_START = 1
EPSILON_END = 0.1
GAMMA = 0.99
ALPHA = 1


def simulate_experience(s, a, task, experiences):

    available_experiences = [x for x in experiences if x[0] ==s and x[1] == a]
    if len(available_experiences) == 0:
        return None, None

    symbols = set(s[1])
    next_task = prog(symbols,task)

    next_s = available_experiences[0][3]

    return next_s,next_task
    
def BFS(state, task, experiences):
    #print(len(experiences))
    QUEUE = []
    extended_state = (state,task)
    EXPLORED_SET = set(extended_state)
    QUEUE.append(extended_state)

    while(QUEUE):
        s,t = QUEUE.pop(0)
        if type(t)== bool:
            if t:
                return True
        
        for a in range(4):
            next_s,next_task = simulate_experience(s, a, t, experiences)

            if next_s != None:
                next_extended_state = (next_s, next_task)
                if next_extended_state not in EXPLORED_SET:
                    EXPLORED_SET.add(next_extended_state)
                    QUEUE.append(next_extended_state)




def main():

    global q
    env = HouseGym()
    
    experience_buffer = set()
    episode_rewards = []
    rollout_rewards = []
    #EPSILON = EPSILON_START

    s, _= env.reset()
    task = env.get_current_task()
    
    for episode in range(N_EPISODES):
        s, _= env.reset()
        symbols = tuple(env.get_symbols())
        extended_state = (s, symbols)

        initial_extended_state = extended_state
        total_reward = 0
        for t in range(N_TIMESTEPS):

            # DELIBERATING ACTION
            a = env.action_space.sample()
            
            ss, reward, done, _ = env.step(a)
            next_symbols = tuple(env.get_symbols())
            next_exteneded_state = (ss, next_symbols)
            total_reward += reward

            #print(extended_state)
            experience_buffer.add((extended_state, a, reward, next_exteneded_state, done))
            extended_state = next_exteneded_state
            
            #simulate_experience(extended_state,1, task, experience_buffer)
            if done:
                break
        
        episode_rewards.append(total_reward)

        rollout_r = BFS(initial_extended_state,task, experience_buffer)
        if rollout_r == None:
            rollout_r = 0
        rollout_rewards.append(rollout_r)
        #print(episode,len(experience_buffer),rollout_r)
        #print(len(experience_buffer))
        if episode%10==0:
            print(f"The running average reward mean of episode {episode} is {np.mean(rollout_rewards[-10:])}")

            
    experiences_count = len(experience_buffer)
    for i in range(1,len(rollout_rewards)):
        convergence = np.mean(rollout_rewards[max(0, i-10):i])
        if convergence == 1:
            print(f"Convergence in {i}, distinct states = {experiences_count}")
            return i, experiences_count
    
    return len(rollout_rewards), experiences_count

if __name__ =='__main__':

    convergence_list = []
    distinct_experiences_list = []
    for i in range(10):
        convergence, distinct_experiences = main()
        #print(convergence_list)
        convergence_list.append(convergence)
        distinct_experiences_list.append(distinct_experiences)

    print(f"TOTAL CONVERGENCE MEAN = {np.mean(convergence_list)}, TOTAL EXPERICENCES MEAN = {np.mean(distinct_experiences_list)}")

