from HouseGym import HouseGym
from HouseGym_2 import HouseGym_2
import numpy as np
import time
import random
import pickle


N_EPISODES = 1000000
N_TIMESTEPS = 100
EPSILON_START = 1
EPSILON_END = 0.1
EPSILON_DECAY_RATE = 0.99999
ALPHA_DECAY_RATE = 0.99999
GAMMA = 0.99
ALPHA = 1

# GLOBAL Q VALUE FUNCTION
q = {}

WORLD_PREDICATES = {'Light', 'Music', 'Monkey'}

## REWARD MACHINE PARAMETERS

RM_STATES = pickle.load(open( 'rm_states.pkl', 'rb'))
RM_TRANSITIONS = pickle.load(open( 'rm_states.pkl', 'rb'))


def Q(state):

    if state not in q:
        q[state] = np.full((4,), 1.0, dtype=np.float64) # initial value
        #q[state] = np.zeros((4,))
    
    return q[state]

def separate_symbols(symbols):

    action_set = symbols.difference(WORLD_PREDICATES)
    world_set = symbols.difference(action_set)

    return action_set, world_set


def main():

    global q, ALPHA
    env = HouseGym_2()
    
    episode_rewards = []
    rollout_rewards = []
    EPSILON = EPSILON_START
    for episode in range(N_EPISODES):
        s, _= env.reset()
        action_set, world_set = separate_symbols(env.get_symbols())
        extended_state = (s, tuple(world_set))
        
        total_reward = 0
        for t in range(N_TIMESTEPS):

            # DELIBERATING ACTION
            a = 0
            if np.random.uniform() <= EPSILON:
                a = env.action_space.sample()
            else:
                a = random.choice([i for i in range(env.action_space.n) if Q(extended_state)[i] == Q(extended_state).max()])
            
            ss, reward, done, _ = env.step(a)
            next_action_set, next_world_set = separate_symbols(env.get_symbols())
            next_exteneded_state = (ss, tuple(next_world_set))
            total_reward += reward
            TD_ERROR = reward +(1-done)* GAMMA*Q(next_exteneded_state).max() - Q(extended_state)[a]

            Q(extended_state)[a] = Q(extended_state)[a] + ALPHA * TD_ERROR
            #print(q)
            extended_state = next_exteneded_state
            
            if done:
                break

        EPSILON = max(EPSILON* EPSILON_DECAY_RATE, EPSILON_END)
        ALPHA = ALPHA*ALPHA_DECAY_RATE
        
        episode_rewards.append(total_reward)
        
        # GREEDY ROLLOUT
        rollout_total_reward = 0
        s, _= env.reset()
        action_set, world_set = separate_symbols(env.get_symbols())
        extended_state = (s, tuple(world_set))
        for t in range(N_TIMESTEPS):
            a = random.choice([i for i in range(env.action_space.n) if Q(extended_state)[i] == Q(extended_state).max()])
            
            ss, reward, done, _ = env.step(a)
            next_action_set, next_world_set = separate_symbols(env.get_symbols())
            next_exteneded_state = (ss, tuple(next_world_set))
            rollout_total_reward += reward
            extended_state = next_exteneded_state

            if done:
                break
        rollout_rewards.append(rollout_total_reward)

        #print(episode, done)
        if episode%1000==0:
            print(f"The running average reward mean of episode {episode} is {np.mean(rollout_rewards[-1000:])}")
            print(f"The number of expanded goals {len(set([x[1] for x in q.keys()]))}")
            #print(q)
            print(f"EPISILON = {EPSILON}")
            print(f'ALPHA = {ALPHA}')

            #print(q)

    print("DONE TRAINING XD")
    print("EXECUTING BEST FOUND POLICY")
    s, _= env.reset()
    action_set, world_set = separate_symbols(env.get_symbols())
    extended_state = (s,  tuple(world_set))
    for t in range(N_TIMESTEPS):
        a = random.choice([i for i in range(env.action_space.n) if Q(extended_state)[i] == Q(extended_state).max()])
        env.render()
        time.sleep(1)
        ss, reward, done, _ = env.step(a)
        next_action_set, next_world_set = separate_symbols(env.get_symbols())
        next_exteneded_state = (ss, tuple(next_world_set))
        rollout_total_reward += reward
        extended_state = next_exteneded_state

        if done:
            break
    
    
    experiences_count = len(q.keys())*4
    for i in range(1,len(rollout_rewards)):
        convergence = np.mean(rollout_rewards[max(0, i-10):i])
        if convergence == 1:
            print(f"Convergence in {i}, distinct states = {experiences_count}")
            q = {}
            return i, experiences_count
    
    q = {}
    return len(rollout_rewards), experiences_count
    



if __name__ =='__main__':
    
    main()
