from HouseGym import HouseGym
import numpy as np
import time
import random


N_EPISODES = 10_000
N_TIMESTEPS = 30
EPSILON_START = 1
EPSILON_END = 0.1
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
        EPSILON = max(EPSILON*0.999, EPSILON_END)
        
        episode_rewards.append(total_reward)
        print(episode, done)
        if episode%100==0:
            print(f"The running average reward mean of episode {episode} is {np.mean(episode_rewards[-10:])}")
            print(f"EPISILON = {EPSILON}")

            #print(q)

            
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
        a = Q(extended_state).argmax()
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
    print(q)
    


if __name__ =='__main__':
    main()
