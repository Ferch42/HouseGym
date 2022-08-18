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

def main():
    env = HouseGym()
    q = np.full((env.size,env.size, env.action_space.n),100.0, dtype = np.float64)
    #q = np.zeros((env.size,env.size, env.action_space.n))

    episode_rewards = []
    EPSILON = EPSILON_START
    for episode in range(N_EPISODES):
        s, _= env.reset()

        total_reward = 0
        for t in range(N_TIMESTEPS):

            # DELIBERATING ACTION
            a = 0
            if np.random.uniform() <= 0:
                a = env.action_space.sample()
            else:
                q_max = q[s[0]][s[1]].max()
                
                actions  = []
                for i in range(env.action_space.n):
                    if q[s[0]][s[1]][i] == q_max:
                        actions.append(i)
                a = random.choice(actions)
                #a = q[s[0]][s[1]].argmax()

            
            ss, reward, done, _ = env.step(a)
            #print(q[4][2][1])

            total_reward += reward
            TD_ERROR = reward +(1-done)* GAMMA*q[ss[0]][ss[1]].max() - q[s[0]][s[1]][a]
            q[s[0]][s[1]][a] = q[s[0]][s[1]][a] + ALPHA * TD_ERROR
            s = ss
            
            if done:
                break
        EPSILON = max(EPSILON*0.999, EPSILON_END)
        
        episode_rewards.append(total_reward)

        if episode%100==0:
            print(f"The running average reward mean of episode {episode} is {np.mean(episode_rewards[-10:])}")
            print(f"EPISILON = {EPSILON}")

            #print(q)

            
    # SIMULATING BEST FOUND POLICY
    time.sleep(10)
    s, _= env.reset()
    env.render()
    time.sleep(1)
    i = 0
    while True:
        
        
        #print(q[s[0]][s[1]])
        a = q[s[0]][s[1]].argmax()
        #time.sleep(2)
        s, reward, done, _ = env.step(a)
        env.render()
        i+=1
        time.sleep(1)
        if i==15 or done:
            break
    
    print(f"The running average reward mean of episode {episode} is {np.mean(episode_rewards[-10:])}")
    for i in range(env.size):
        for j in range(env.size):
            print(i,j)
            print(q[i][j])
    print(q)

    


if __name__ =='__main__':
    main()
