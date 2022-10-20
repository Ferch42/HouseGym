world = (('NOT','Light'),('NOT','Music'), ('NOT', 'Monkey'))
ACTIONS = {'TRUE', 'LightSwitch', 'Radio', 'Ball'}


def invert_predicate(P):

    if type(P) == tuple:
        return P[1]
    elif type(P) == str:
        return ('NOT', P)



def expand_world_into_formula(world):

    LIGHT, MUSIC, MONKEY = world

    return ('AND', LIGHT, ('AND', MUSIC,  MONKEY))

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


def generate_formula(WORLD):

    GOAL, PARENT_TREE =bfs(WORLD)
    W = GOAL 
    
    formula = expand_world_into_formula(GOAL)

    while W!= WORLD:

        print(W)
        print(PARENT_TREE[W])
        PREV_WORLD, ACTION = PARENT_TREE[W]
        
        formula = ('UNTIL', expand_world_into_formula(PREV_WORLD), ('AND', ACTION, ('NEXT', formula)))
        W = PREV_WORLD
    #formula = ('UNTIL', expand_world_into_formula(PREV_WORLD), ('AND', ACTION, ('NEXT', formula)))
    return formula


if __name__== "__main__":

    world = ('Light',('NOT','Music'), ('NOT', 'Monkey'))
    print(world)
    print(advance_world(world, 'LightSwitch'))
    print(advance_world(world, 'Radio'))
    world_2 = (('NOT', 'Light'), 'Music', ('NOT', 'Monkey'))
    print(world_2)
    print(advance_world(world_2, 'Ball'))
    print('________________________________________________________________________________')

    INITIAL_W = ('Light',('NOT','Music'), ('NOT', 'Monkey'))

    print("__________________________________________________________________")
    print(generate_formula(INITIAL_W))
    print("__________________________________________________________________")