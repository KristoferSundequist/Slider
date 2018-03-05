from multiprocessing import Pool
import numpy as np
import math
import slider
import time
	
np.random.seed(0)

# RMSprop decay
decay = 0.9

# l2-loss param
weight_decay=0.01

# Network params
n_inputs = 8
n_hidden = 200
n_hidden2 = 200
n_out = 4;

# Init weights
current_b1 = np.zeros((1,n_hidden))
current_w1 = np.random.normal(0, np.sqrt(2.0/(n_inputs+n_hidden)), (n_inputs,n_hidden))
current_b2 = np.zeros((1,n_hidden2))
current_w2 = np.random.normal(0, np.sqrt(2.0/(n_hidden+n_hidden2)), (n_hidden,n_hidden2))
current_b3 = np.zeros((1,n_out))
current_w3 = np.random.normal(0, np.sqrt(2.0/(n_hidden2+n_out)), (n_hidden2,n_out))

# RMSProp
rmsprop_b1 = np.zeros_like(current_b1)
rmsprop_w1 = np.zeros_like(current_w1)
rmsprop_b2 = np.zeros_like(current_b2)
rmsprop_w2 = np.zeros_like(current_w2)
rmsprop_b3 = np.zeros_like(current_b3)
rmsprop_w3 = np.zeros_like(current_w3)

# Forward propgation using provided weights with softmax output
def forward(b1,w1,b2,w2,b3,w3,inp):
    h = inp.dot(w1) + b1
    h[h<0] = 0
    h2 = h.dot(w2) + b2
    h2[h2<0] = 0
    out = h2.dot(w3) + b3
    return np.exp(out) / (np.sum(np.exp(out)) + 1e-20)

# Samples an action using weights and given state
def get_action(b1,w1,b2,w2,b3,w3,state):
    dist = forward(b1,w1,b2,w2,b3,w3,state).reshape(4)
    return np.argmax(dist)
    #return np.random.choice(4, 1, p=dist)[0]

# Sample action given state using current weights (wrapper)
def get_action_current(state):
    return get_action(current_b1, current_w1, current_b2, current_w2, current_b3, current_w3, state)

def l2loss(b1,w1,b2,w2,b3,w3):
    return weight_decay*(np.mean(b1*b1) + np.mean(w1*w1) + np.mean(b2*b2) + np.mean(w2*w2) + np.mean(b3*b3) + np.mean(w3*w3))

# Eval some weights for iters steps (episode size)
def fitness(b1,w1,b2,w2,b3,w3,iters):
    slider.reset()
    accReward = 0
    state = slider.get_state()
    for _ in range(iters):
        action = get_action(b1,w1,b2,w2,b3,w3,state)
        reward,state = slider.step(action)
        accReward += reward
        
    return accReward #- l2loss(b1,w1,b2,w2,b3,w3)

# Eval current weights in env (wrapper)
def fitness_current(iters):
    return fitness(current_b1, current_w1, current_b2, current_w2, current_b3, current_w3, iters)

# Watch agent do its thing
def agent_loop(iters):
    slider.reset()
    for _ in range(iters):
        state = slider.get_state()
        action = get_action_current(state)
        _,_ = slider.step(action)
        slider.render()
        time.sleep(0.01)

# Dummy env to make sure stuff works (sanity check)
def fitness2(b1,w1,b2,w2,b3,w3):
    a = forward(b1,w1,b2,w2,b3,w3,np.array([1,1,1,1,1,1,1,1])).reshape(4)
    b = forward(b1,w1,b2,w2,b3,w3,np.array([1,0,1,-5,14,1,0,1])).reshape(4)
    c = forward(b1,w1,b2,w2,b3,w3,np.array([0,1,0,1,0,1,0,1])).reshape(4)
    d = forward(b1,w1,b2,w2,b3,w3,np.array([1,1,4,1,-1,1,11,0])).reshape(4)
    e = forward(b1,w1,b2,w2,b3,w3,np.array([0,1,-3,1,1,5,1,1])).reshape(4)
    f = forward(b1,w1,b2,w2,b3,w3,np.array([0,0,0,0,1,5,1,-2])).reshape(4)
    g = forward(b1,w1,b2,w2,b3,w3,np.array([1,0,1,0,-3,0,0,1])).reshape(4)
    h = forward(b1,w1,b2,w2,b3,w3,np.array([-10,-2,0,-2,0,1,0,20])).reshape(4)
    i = forward(b1,w1,b2,w2,b3,w3,np.array([1,3,3,1,3,3,3,0])).reshape(4)
    j = forward(b1,w1,b2,w2,b3,w3,np.array([11,1,-3,0,0,10,-7,-3])).reshape(4)
    
    #return -(1.0 - a[3] + 1 - b[1] + int(c == 0) + int(d == 2) + int(e == 3))
    return a[3] + b[1] + c[0] + d[2] + e[3] + f[3] + g[2] + h[0] + i[1] + j[0]

# harder dummy
def fitness3(b1,w1,b2,w2,b3,w3):
    a = np.random.choice(4, 1, p=forward(b1,w1,b2,w2,b3,w3,np.array([1,1,1,1,1,1,1,1])).reshape(4))[0]
    b = np.random.choice(4, 1, p=forward(b1,w1,b2,w2,b3,w3,np.array([1,0,1,-5,14,1,0,1])).reshape(4))[0]
    c = np.random.choice(4, 1, p=forward(b1,w1,b2,w2,b3,w3,np.array([0,1,0,1,0,1,0,1])).reshape(4))[0]
    d = np.random.choice(4, 1, p=forward(b1,w1,b2,w2,b3,w3,np.array([1,1,4,1,-1,1,11,0])).reshape(4))[0]
    e = np.random.choice(4, 1, p=forward(b1,w1,b2,w2,b3,w3,np.array([0,1,-3,1,1,5,1,1])).reshape(4))[0]
    f = np.random.choice(4, 1, p=forward(b1,w1,b2,w2,b3,w3,np.array([0,0,0,0,1,5,1,-2])).reshape(4))[0]
    g = np.random.choice(4, 1, p=forward(b1,w1,b2,w2,b3,w3,np.array([1,0,1,0,-3,0,0,1])).reshape(4))[0]
    h = np.random.choice(4, 1, p=forward(b1,w1,b2,w2,b3,w3,np.array([-10,-2,0,-2,0,1,0,20])).reshape(4))[0]
    i = np.random.choice(4, 1, p=forward(b1,w1,b2,w2,b3,w3,np.array([1,3,3,1,3,3,3,0])).reshape(4))[0]
    j = np.random.choice(4, 1, p=forward(b1,w1,b2,w2,b3,w3,np.array([11,1,-3,0,0,10,-7,-3])).reshape(4))[0]

    return int(a==3) + int(b==1) + int(c==0) + int(d==2) + int(e==3) + int(f==3) + int(g==2) + int(h==0) + int(i==1) + int(j==0)

# deterministic dummy
def fitness4(b1,w1,b2,w2,b3,w3):
    a = np.argmax(forward(b1,w1,b2,w2,b3,w3,np.array([1,1,1,1,1,1,1,1])))
    b = np.argmax(forward(b1,w1,b2,w2,b3,w3,np.array([1,0,1,-5,14,1,0,1])))
    c = np.argmax(forward(b1,w1,b2,w2,b3,w3,np.array([0,1,0,1,0,1,0,1])))
    d = np.argmax(forward(b1,w1,b2,w2,b3,w3,np.array([1,1,4,1,-1,1,11,0])))
    e = np.argmax(forward(b1,w1,b2,w2,b3,w3,np.array([0,1,-3,1,1,5,1,1])))
    f = np.argmax(forward(b1,w1,b2,w2,b3,w3,np.array([0,0,0,0,1,5,1,-2])))
    g = np.argmax(forward(b1,w1,b2,w2,b3,w3,np.array([1,0,1,0,-3,0,0,1])))
    h = np.argmax(forward(b1,w1,b2,w2,b3,w3,np.array([-10,-2,0,-2,0,1,0,20])))
    i = np.argmax(forward(b1,w1,b2,w2,b3,w3,np.array([1,3,3,1,3,3,3,0])))
    j = np.argmax(forward(b1,w1,b2,w2,b3,w3,np.array([11,1,-3,0,0,10,-7,-3])))

    return int(a==3) + int(b==1) + int(c==0) + int(d==2) + int(e==3) + int(f==3) + int(g==2) + int(h==0) + int(i==1) + int(j==0)# - l2loss(b1,w1,b2,w2,b3,w3)

# export OMP_NUM_THREADS=1

#train(100,0.1,0.01,30,1000)
# Main training function
# ------------------------
# npop: Size of population
# sigma: standard deviation of population
# alpha: learning rate
# episodelengths: number of steps in environment
# workers: number of workers
def train(npop, sigma, alpha, gen, episodelengths, workers):
    pool = Pool(processes=workers)
    
    global current_b1
    global current_w1
    global current_b2
    global current_w2
    global current_b3
    global current_w3

    global rmsprop_b1
    global rmsprop_w1
    global rmsprop_b2
    global rmsprop_w2
    global rmsprop_b3
    global rmsprop_w3

    mr = 0
    start = time.time()
    for i in range(gen):

        if i % 5 == 0:
            print(i, " ", fitness(current_b1, current_w1, current_b2, current_w2, current_b3, current_w3, episodelengths), " sigma: ", sigma)
            #print(i, " ", fitness4(current_b1, current_w1, current_b2, current_w2, current_b3, current_w3), " sigma: ", sigma, " alpha: ", alpha)
            #sigma = sigma*0.99
            #alpha = alpha*0.99

        # Create random permutations for weights
        snpop = int(npop/2)
        dist = [0.7,0.3]
        N_b1 = sigma*np.random.randn(snpop, 1, n_hidden)/n_hidden
        N_b1 *= np.random.choice(2,(snpop, 1, n_hidden),p=dist)
        #N_b1 *= np.random.randint(2, size=(snpop, 1, n_hidden))
        N_b1 = np.concatenate((N_b1, -N_b1))

        N_w1 = sigma*np.random.randn(snpop, n_inputs, n_hidden)
        #N_w1 *= np.random.randint(2, size=(snpop, n_inputs, n_hidden))
        N_w1 *= np.random.choice(2,(snpop, n_inputs, n_hidden),p=dist)
        N_w1 = np.concatenate((N_w1, -N_w1))
        
        N_b2 = sigma*np.random.randn(snpop, 1, n_hidden2)/n_hidden2
        #N_b2 *= np.random.randint(2, size=(snpop, 1, n_hidden2))
        N_b2 *= np.random.choice(2,(snpop, 1, n_hidden2),p=dist)
        N_b2 = np.concatenate((N_b2, -N_b2))
        
        N_w2 = sigma*np.random.randn(snpop, n_hidden, n_hidden2)
        #N_w2 *= np.random.randint(2, size=(snpop, n_hidden, n_hidden2))
        N_w2 *= np.random.choice(2,(snpop, n_hidden, n_hidden2),p=dist)
        N_w2 = np.concatenate((N_w2, -N_w2))
        
        N_b3 = sigma*np.random.randn(snpop, 1, n_out)/n_out
        #N_b3 *= np.random.randint(2, size=(snpop, 1, n_out))
        N_b3 *= np.random.choice(2,(snpop, 1, n_out),p=dist)
        N_b3 = np.concatenate((N_b3, -N_b3))
        
        N_w3 = sigma*np.random.randn(snpop, n_hidden2, n_out)
        #N_w3 *= np.random.randint(2, size=(snpop, n_hidden2, n_out))
        N_w3 *= np.random.choice(2,(snpop, n_hidden2, n_out),p=dist)
        N_w3 = np.concatenate((N_w3, -N_w3))
        
        # Create population of nets
        nets = [(current_b1 + N_b1[j],
                 current_w1 + N_w1[j],
                 current_b2 + N_b2[j],
                 current_w2 + N_w2[j],
                 current_b3 + N_b3[j],
                 current_w3 + N_w3[j],
                 episodelengths
                 )
                for j in range(npop)]


        # Calculate fitnesses of nets
        R = np.array(pool.starmap(fitness, nets))
        
        # Print mean reward
        if i == 0:
            mr = np.mean(R)
        else:
            mr = 0.9*mr + (0.1)*np.mean(R)
        print(mr)

        # Rank transform fitnesses
        t = R.argsort()
        ranks = np.empty_like(t)
        ranks[t] = np.arange(len(R))
        A = (ranks - np.mean(ranks)) / (np.std(ranks)*3 + 0.000001)

        # Calc weight updates based on fitnesses
        delta_b1 =  1/(npop) * np.dot(N_b1.T, A).T
        delta_w1 =  1/(npop) * np.dot(N_w1.T, A).T
        delta_b2 =  1/(npop) * np.dot(N_b2.T, A).T
        delta_w2 =  1/(npop) * np.dot(N_w2.T, A).T
        delta_b3 =  1/(npop) * np.dot(N_b3.T, A).T
        delta_w3 =  1/(npop) * np.dot(N_w3.T, A).T

        # Update RMSProp
        rmsprop_b1 = decay*rmsprop_b1 + (1-decay)*(delta_b1**2)
        rmsprop_w1 = decay*rmsprop_w1 + (1-decay)*(delta_w1**2)
        rmsprop_b2 = decay*rmsprop_b2 + (1-decay)*(delta_b2**2)
        rmsprop_w2 = decay*rmsprop_w2 + (1-decay)*(delta_w2**2)
        rmsprop_b3 = decay*rmsprop_b3 + (1-decay)*(delta_b3**2)
        rmsprop_w3 = decay*rmsprop_w3 + (1-decay)*(delta_w3**2)

        # Update weights
        current_b1 += alpha * delta_b1/(np.sqrt(rmsprop_b1) + 1e-5)
        current_w1 += alpha * delta_w1/(np.sqrt(rmsprop_w1) + 1e-5)
        current_b2 += alpha * delta_b2/(np.sqrt(rmsprop_b2) + 1e-5)
        current_w2 += alpha * delta_w2/(np.sqrt(rmsprop_w2) + 1e-5)
        current_b3 += alpha * delta_b3/(np.sqrt(rmsprop_b3) + 1e-5)
        current_w3 += alpha * delta_w3/(np.sqrt(rmsprop_w3) + 1e-5)

    end = time.time()
    print(end-start)
    pool.close()
