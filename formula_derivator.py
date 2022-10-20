world = (('NOT','Light'),('NOT','Music'), ('NOT', 'Monkey'))
ACTIONS = {'TRUE', 'LightSwitch', 'Radio', 'Ball'}


def invert_predicate(P):

    if type(P) == tuple:
        return P[1]
    elif type(P) == str:
        return ('NOT', P)


def advance_world(world, action):

    LIGHT, MUSIC , MONKEY = world
    if action == 'TRUE':
        return world
    
    elif action =='LightSwitch':
        return (invert_predicate(LIGHT), MUSIC, MONKEY)
        
    elif action == 'Radio' and 'Light' in world:
        return (LIGHT, invert_predicate(MUSIC), MONKEY)

    elif action == 'Ball' and 'Music' in world and ('NOT', 'Light') in world:
        return (LIGHT, MUSIC, invert_predicate(MONKEY))

    else:
        return world




def bfs(world):

    QUEUE = []
    EXPLORED_SET = {world}
    QUEUE.append(world)
    PARENT = {}
    
    while QUEUE:
        W = QUEUE.pop(0)
        
        if 'Monkey' in W:
            return W, PARENT
        
        for A in ACTIONS:
            NEXT_W = advance_world(W, A)
            
            if NEXT_W not in EXPLORED_SET:
                PARENT[NEXT_W] = (W, A)
                EXPLORED_SET.add(NEXT_W)
                QUEUE.append(NEXT_W)

    


if __name__== "__main__":

    world = ('Light',('NOT','Music'), ('NOT', 'Monkey'))
    print(world)
    print(advance_world(world, 'LightSwitch'))
    print(advance_world(world, 'Radio'))
    world_2 = (('NOT', 'Light'), 'Music', ('NOT', 'Monkey'))
    print(world_2)
    print(advance_world(world_2, 'Ball'))
    print('________________________________________________________________________________')

    GOAL, PARENT =bfs((('NOT','Light'),('NOT','Music'), ('NOT', 'Monkey')))
    
    print('________________________________________________________________________________')

    GOAL, PARENT = bfs(('Light',('NOT','Music'), ('NOT', 'Monkey')))

    W = GOAL 
    INITIAL_W = ('Light',('NOT','Music'), ('NOT', 'Monkey'))
    while W!= INITIAL_W:

        print(f"W: {W}")
        print(f"PARENT : {PARENT[W]}")
        W = PARENT[W]