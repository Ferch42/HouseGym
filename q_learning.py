from HouseGym import HouseGym
import numpy as np
import time


N_EPISODES = 1_000
N_TIMESTEPS = 30
EPSILON_START = 1
EPSILON_END = 0.1
GAMMA = 0.99
ALPHA = 0.1

def main():
    env = HouseGym()
    q = np.zeros(shape= (env.size,env.size, env.action_space.n))
    
    episode_rewards = []
    EPSILON = EPSILON_START
    for episode in range(N_EPISODES):
        s, _= env.reset()

        total_reward = 0
        for t in range(N_TIMESTEPS):

            # DELIBERATING ACTION
            a = 0
            if np.random.uniform() <= EPSILON:
                a = env.action_space.sample()
            else:
                a = q[s[0]][s[1]].argmax()

            
            ss, reward, done, _ = env.step(a)
            total_reward += reward
            TD_ERROR = reward + GAMMA*q[ss[0]][ss[1]].max() - q[s[0]][s[1]][a]
            q[s[0]][s[1]][a] = q[s[0]][s[1]][a] + ALPHA * TD_ERROR
            s = ss
            if done:
                break
        EPSILON = max(EPSILON*0.999, EPSILON_END)
        
        episode_rewards.append(total_reward)

        if episode%100==0:
            print(f"The running average reward mean of episode {episode} is {np.mean(episode_rewards[-10:])}")
            #print(f"EPISILON = {EPSILON}")

            #print(q)

            
    # SIMULATING BEST FOUND POLICY
    s, _= env.reset()
    env.render()
    time.sleep(1)
    while True:

        a = q[s[0]][s[1]].argmax()
        
        s, reward, done, _ = env.step(a)
        env.render()
        time.sleep(1)
        if done:
            break
    print(f"The running average reward mean of episode {episode} is {np.mean(episode_rewards[-10:])}")

    


if __name__ =='__main__':
    main()
