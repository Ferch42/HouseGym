from HouseGym import HouseGym
from HouseGym_2 import HouseGym_2
import numpy as np
import time
import random
import pickle


N_EPISODES = 10000
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
ACTIONS = {'Nothing', 'LightSwitch', 'Radio', 'Ball'}

MODE = 'Model'

## REWARD MACHINE PARAMETERS

RM_STATES = pickle.load(open( 'rm_states.pkl', 'rb'))
RM_TRANSITIONS = pickle.load(open( 'rm_transitions.pkl', 'rb'))

print(RM_STATES)
print(RM_TRANSITIONS)

def Q(state):

    if state not in q:
        q[state] = np.full((4,), 1.0, dtype=np.float64) # initial value
        #q[state] = np.zeros((4,))
    
    return q[state]

def separate_symbols(symbols):

    action_set = symbols.difference(WORLD_PREDICATES)
    world_set = symbols.difference(action_set)

    return action_set, tuple(sorted(world_set))

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
                    

def RMAX(experiences):
    
    STATES, TRANSITION_MATRIX, REWARD_FUNCTION = get_model(experiences)
    V = {s: 0 for s in STATES}
    P = {s: 0 for s in STATES}
    
    
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
            #print(value_delta)
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
    
    Q_ = {s: np.zeros((4,)) for s in STATES}
    for s in STATES:
        for a in range(4):
            ns_set = set(x for x in TRANSITION_MATRIX[s][a])
            Q_[s][a] = sum(TRANSITION_MATRIX[s][a][ns]*(REWARD_FUNCTION[s][a][ns] + GAMMA* V[ns]) for ns in ns_set)

    return Q_



def main():

    global q, ALPHA
    env = HouseGym_2()
    
    episode_rewards = []
    rollout_rewards = []
    EPSILON = EPSILON_START
    experience_buffer = []

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
            assert(next_world_set in RM_STATES)
            next_exteneded_state = (ss, tuple(next_world_set))
            total_reward += reward
            
            if MODE == "Normal":
                TD_ERROR = reward +(1-done)* GAMMA*Q(next_exteneded_state).max() - Q(extended_state)[a]
                Q(extended_state)[a] = Q(extended_state)[a] + ALPHA * TD_ERROR
            
            elif MODE == "CRM":

                original_s, original_w = extended_state
                #print(original_s)
                symbolic_action_set = next_action_set.difference({'Nothing'})
                if len(symbolic_action_set)==1:
                    #print(symbolic_action_set)
                    symbolic_action = list(symbolic_action_set)[0]
                else:
                    symbolic_action = 'Nothing'

                assert(len(symbolic_action_set)<=1)
                for RM_S in RM_STATES:
                    
                    new_rm_state = (original_s, RM_S)
                    rm_transition = [x for x in RM_TRANSITIONS if x[0] == RM_S and x[1] == symbolic_action]
                    assert(len(rm_transition)==1)
                    NEXT_RM_S = rm_transition[0][2]
                    rm_reward = int('Monkey' in NEXT_RM_S)
                    #print(NEXT_RM_S,rm_transition,rm_reward)
                    
                    rm_done = int('Monkey' in NEXT_RM_S)
                    next_rm_state = (ss, NEXT_RM_S)
                    

                    TD_ERROR = rm_reward +(1-rm_done)* GAMMA*Q(next_rm_state).max() - Q(new_rm_state)[a]
                    
                    #print('------------------------------------------------')
                    #print(new_rm_state, symbolic_action, rm_reward, next_rm_state)
                    #print(new_rm_state, Q(new_rm_state)[a], TD_ERROR)
                    Q(new_rm_state)[a] = Q(new_rm_state)[a] + ALPHA * TD_ERROR


            elif MODE=="Model":

                original_s, original_w = extended_state
                #print(original_s)
                symbolic_action_set = next_action_set.difference({'Nothing'})
                if len(symbolic_action_set)==1:
                    #print(symbolic_action_set)
                    symbolic_action = list(symbolic_action_set)[0]
                else:
                    symbolic_action = 'Nothing'

                assert(len(symbolic_action_set)<=1)
                for RM_S in RM_STATES:
                    
                    new_rm_state = (original_s, RM_S)
                    rm_transition = [x for x in RM_TRANSITIONS if x[0] == RM_S and x[1] == symbolic_action]
                    assert(len(rm_transition)==1)
                    NEXT_RM_S = rm_transition[0][2]
                    rm_reward = int('Monkey' in NEXT_RM_S)
                    #print(NEXT_RM_S,rm_transition,rm_reward)
                    
                    rm_done = int('Monkey' in NEXT_RM_S)
                    next_rm_state = (ss, NEXT_RM_S)
                    
                    experience_buffer.append((new_rm_state, a, rm_reward, next_rm_state, rm_done))
                    experience_buffer = experience_buffer[-10_000:]

                    #TD_ERROR = rm_reward +(1-rm_done)* GAMMA*Q(next_rm_state).max() - Q(new_rm_state)[a]
                    
                    #print('------------------------------------------------')
                    #print(new_rm_state, symbolic_action, rm_reward, next_rm_state)
                    #print(new_rm_state, Q(new_rm_state)[a], TD_ERROR)
                    #Q(new_rm_state)[a] = Q(new_rm_state)[a] + ALPHA * TD_ERROR
                #print(f'halo+{t}')
                q = RMAX(experience_buffer)
                #print(q)


            else:
                raise Exception("Not valid option")


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
            print(f'The size of the state-action space is {len(q.keys())}')
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
    
    
    #with open('reward_machine.txt', 'w+') as f:
    #    f.write(str(q.keys()))

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
