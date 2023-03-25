import slider
import time
import memory
import random
import math
import nstep
import policy
import globals

agent = policy.Policy()

replay_memory_size = 100000
replay_memory = memory.ReplayMemory(replay_memory_size)

# export OMP_NUM_THREADS=1
def live(iterations, lagg, eps, improve_flag):
    n_step = nstep.Nstep(globals.num_steps)
    g = slider.Game()
    state = g.get_state()
    total_reward = 0
    start = time.time()
    for i in range(iterations):

        # eps-greedy
        if random.uniform(0,1) < eps:
            action = random.randint(0,3)
        else:
            q_vals = agent.get_Q_values(state)
            action = q_vals.argmax()
            
        reward,next_state = g.step(action)
        
        n_step.push(state,action,reward)
        
        state = next_state
        
        if i >= globals.num_steps:
            replay_memory.push(n_step.get())

        if i % lagg == 0 and improve_flag:
            agent.update(replay_memory.get_memories())

        total_reward += reward

    end = time.time()
    print("Elapsed time: ", end-start)
    print(total_reward)

def init_memory(iters=replay_memory_size, eps=1):
    live(iters, 9999, eps, False)
    print("Memory initiated")
    agent.init_agent(replay_memory.get_memories())
    print("Memory fitted")
    
def live_loop(lives, iterations, lagg, eps):
    for i in range(lives):
        print(f'Iteration: {i} of {lives}')
        live(iterations, lagg, eps, True)

def agent_loop(iterations, cd):
    g = slider.Game()
    g.render(0,0,0,0)
    for i in range(iterations):
        state = g.get_state()
        q_vals = agent.get_Q_values(state)
        _,_ = g.step(q_vals.argmax())        
        g.render(round(q_vals[0], 4), round(q_vals[1], 4), round(q_vals[2], 4), round(q_vals[3], 4))
        g.render(0,0,0,0)
        time.sleep(cd)

init_memory()

