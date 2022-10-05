from HouseGym import HouseGym
import numpy as np
import time
import random
from LTL import *


N_EPISODES = 100
N_TIMESTEPS = 30
#EPSILON_START = 1
#EPSILON_END = 0.1
GAMMA = 0.99
EXPERIENCE_BUFFER_SIZE = 1_000
#ALPHA = 1


def get_model(experiences):

    STATES = set(x[0] for x in experiences).union(set(x[3] for x in experiences))
    TRANSITION_MATRIX = {}  
    REWARD_FUNCTION = {}

    for s in STATES:
        TRANSITION_MATRIX[s] = {}
        REWARD_FUNCTION[s] = {}
        for a in range(4): # ACTION_SPACE
            available_experiences = [x for x in experiences if x[0] ==s and x[1] == a]
            if len(available_experiences) == 0:
                TRANSITION_MATRIX[s][a] = {'ABSORBING_STATE' : 1}
                REWARD_FUNCTION[s][a] =  {'ABSORBING_STATE' : 1}
            else:
                next_states_set = set(x[3] for x in available_experiences)
                number_of_experiences = len(available_experiences)
                TRANSITION_MATRIX[s][a]  = {}
                REWARD_FUNCTION[s][a]  = {}
                for ns in next_states_set:
                    next_state_experiences = [x for x in available_experiences if x[3] == ns]
                    mean_reward = np.mean([x[2] for x in next_state_experiences])
                    TRANSITION_MATRIX[s][a][ns] = len(next_state_experiences)/ number_of_experiences # ESTIMATE TRANSITION PROB
                    REWARD_FUNCTION[s][a][ns] =  mean_reward

    STATES.add('ABSORBING_STATE')
    TRANSITION_MATRIX['ABSORBING_STATE'] = {x: {'ABSORBING_STATE': 1} for x in range(4)}
    REWARD_FUNCTION['ABSORBING_STATE'] = {x: {'ABSORBING_STATE': 0} for x in range(4)}
    
    return STATES, TRANSITION_MATRIX, REWARD_FUNCTION   
                    


def RMAX(extended_state, experiences):
    
    STATES, TRANSITION_MATRIX, REWARD_FUNCTION = get_model(experiences)
    V = {s: 0 for s in STATES}
    P = {s: 0 for s in STATES}

    if extended_state not in STATES:
        return np.random.randint(4)
    
    # POLICY ITERATION
    while True:

        # POLICY EVALUATION
        while True:
            value_delta = 0
            for s in STATES:
                a = P[s]
                ns_set = set(x for x in TRANSITION_MATRIX[s][a])
                v = V[s]
                V[s] = sum(TRANSITION_MATRIX[s][a][ns]*(REWARD_FUNCTION[s][a][ns] + GAMMA* V[ns]) for ns in ns_set)
                value_delta = max(value_delta, abs(V[s]-v))
                #time.sleep(2)
            if value_delta < 0.00001:
                break

        # POLICY IMPROVEMENT
        policy_stable = True
        for s in STATES:
            old_action = P[s]
            P[s] = np.argmax([sum(TRANSITION_MATRIX[s][a][ns]*(REWARD_FUNCTION[s][a][ns] + GAMMA*V[ns]) for ns in set(x for x in TRANSITION_MATRIX[s][a])) for a in range(4)])
            if old_action != P[s]:
                policy_stable = False
        
        if policy_stable:
            break

    return P[extended_state]




def main():

    env = HouseGym()
    
    experience_buffer = list()
    episode_rewards = []
    #EPSILON = EPSILON_START

    s, _= env.reset()
    task = env.get_current_task()
    
    for episode in range(N_EPISODES):
        s, _= env.reset()
        task = env.get_current_task()
        extended_state = (s, task)

        initial_extended_state = extended_state
        total_reward = 0
        for t in range(N_TIMESTEPS):

            # DELIBERATING ACTION
            a = RMAX(extended_state, experience_buffer)
            
            ss, reward, done, _ = env.step(a)
            next_task = env.get_current_task()
            next_exteneded_state = (ss, next_task)
            total_reward += reward

            experience_buffer.append((extended_state, a, reward, next_exteneded_state, done))
            experience_buffer = experience_buffer[-EXPERIENCE_BUFFER_SIZE:]
            extended_state = next_exteneded_state

            if done:
                break
        
        episode_rewards.append(total_reward)

        #print(episode,len(experience_buffer),BFS(initial_extended_state,task, experience_buffer))
        #print(len(experience_buffer))
        if episode%10==0:
            print(f"The running average reward mean of episode {episode} is {np.mean(episode_rewards[-10:])}")

            
    # SIMULATING BEST FOUND POLICY
    time.sleep(10)
    
    s, _= env.reset()
    task = env.get_current_task()
    extended_state = (s, task)
    env.render()
    time.sleep(1)
    i = 0
    while True:
        
        
        #print(q[s[0]][s[1]])
        a = RMAX(extended_state, experience_buffer)
        #time.sleep(2)
        s, reward, done, _ = env.step(a)
        task = env.get_current_task()
        extended_state = (s,task)
        env.render()
        i+=1
        time.sleep(1)
        if i==30 or done:
            break
    
    print(f"The running average reward mean of episode {episode} is {np.mean(episode_rewards[-10:])}")
    


if __name__ =='__main__':
    main()
