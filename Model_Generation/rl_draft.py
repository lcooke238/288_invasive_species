# reinforcement learning 
# reward is -(difference in density of predators)
# placing a trap has -1 reward, not placing a trap has no reward 
# decreasing the density of all predators to 0 is reward + 30 
# state-action space is grid of traps x grid of densities, discretized 

# traps and density are both n x m grids 
# returns next state 
from collections import defaultdict 
import random 
import synthetic_data

ALPHA = 0.2 
GAMMA = 0.5
EPSILON = 0.1
M = 20 
N = 20 
Q_TABLE = defaultdict(float)
DONE = False 

def LV(densities, traps): 
    # this function should return the density of the predator given that the initial density of the predator was densities and we placed traps 
    
def next_state(state, action): 
    traps, prev = state[0], state[1]
    row, col = action 
    traps[row][col] = 1 - traps[row][col]
    next_ = LV(prev, traps)
    return (traps, next_)

def reward(state, next_state): 
    r = np.sum(state[1] - next_state[1])
    if np.sum(next_state[1]) < 1: 
        DONE = True 
        return 20 
    return r 

def maximum_over_actions(state): 
    best = 0 
    for row in range(M): 
        for col in range(N): 
            best = max(best, Q_TABLE((state, (row, col))))
    return best

def train_agent(start_state): 
    state = start_state
    for epoch in range(1000): 
        if random.uniform(0, 1) < EPSILON: 
            action = (random.randint(0, M - 1), random.randint(0, N - 1))
        else: 
            action = np.argmax()
        next_ = next_state(state, action) 
        r = reward(state, next_)
        Q_TABLE[(state, action)] = (1 - ALPHA) * Q_TABLE[(state, action)] + ALPHA * (reward + GAMMA * maximum_over_actions(next_state))
        state = next_ 
        if DONE == True:
            break 

# return the predator density and ??? 
def reset(): 
    data_init = synthetic_data.generate(m=m, n=n)
    self.predator_density = data_init[:, :, 1, :]
    traps = np.zeros((m, n))
    return (self.predator_density, traps)
