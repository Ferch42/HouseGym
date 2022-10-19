from HouseGym import HouseGym
import numpy as np
import time
import random


N_EPISODES = 1000
N_TIMESTEPS = 30
EPSILON_START = 1
EPSILON_END = 0.1
EPSILON_DECAY_RATE = 0.9
GAMMA = 0.99
ALPHA = 1

# GLOBAL Q VALUE FUNCTION
q = {}

def Q(state):

    if state not in q:
        q[state] = np.full((4,), 1.0, dtype=np.float64) # initial value
        #q[state] = np.zeros((4,))
    
    return q[state]

def main():

    global q
    env = HouseGym()
    
    episode_rewards = []
    rollout_rewards = []
    EPSILON = EPSILON_START
    for episode in range(N_EPISODES):
        s, _= env.reset()
        task = env.get_current_task()
        extended_state = (s, task)

        total_reward = 0
        for t in range(N_TIMESTEPS):

            # DELIBERATING ACTION
            a = 0
            if np.random.uniform() <= 0:
                a = env.action_space.sample()
            else:
                a = random.choice([i for i in range(env.action_space.n) if Q(extended_state)[i] == Q(extended_state).max()])
            
            ss, reward, done, _ = env.step(a)
            next_task = env.get_current_task()
            next_exteneded_state = (ss, next_task)
            total_reward += reward
            TD_ERROR = reward +(1-done)* GAMMA*Q(next_exteneded_state).max() - Q(extended_state)[a]

            Q(extended_state)[a] = Q(extended_state)[a] + ALPHA * TD_ERROR
            #print(q)
            extended_state = next_exteneded_state
            
            if done:
                break
        EPSILON = max(EPSILON* EPSILON_DECAY_RATE, EPSILON_END)
        
        episode_rewards.append(total_reward)
        
        # GREEDY ROLLOUT
        rollout_total_reward = 0
        s, _= env.reset()
        task = env.get_current_task()
        extended_state = (s, task)
        for t in range(N_TIMESTEPS):
            a = random.choice([i for i in range(env.action_space.n) if Q(extended_state)[i] == Q(extended_state).max()])
            
            ss, reward, done, _ = env.step(a)
            next_task = env.get_current_task()
            next_exteneded_state = (ss, next_task)
            rollout_total_reward += reward
            extended_state = next_exteneded_state

            if done:
                break
        rollout_rewards.append(rollout_total_reward)

        #print(episode, done)
        #if episode%10==0:
            #print(f"The running average reward mean of episode {episode} is {np.mean(rollout_rewards[-10:])}")
            #print(f"EPISILON = {EPSILON}")

            #print(q)
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
    
    convergence_list = []
    distinct_experiences_list = []
    for i in range(30):
        convergence, distinct_experiences = main()
        #print(convergence_list)
        convergence_list.append(convergence)
        distinct_experiences_list.append(distinct_experiences)

    print(f"TOTAL CONVERGENCE MEAN = {np.mean(convergence_list)}, TOTAL EXPERICENCES MEAN = {np.mean(distinct_experiences_list)}")
