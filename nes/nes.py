import numpy as np
import math
import slider
import time
	
#np.random.seed(0)


n_inputs = 8
n_hidden = 200
n_hidden2 = 100
n_out = 4;

current_b1 = np.zeros((1,n_hidden))
current_w1 = np.random.normal(0, np.sqrt(2.0/(n_inputs+n_hidden)), (n_inputs,n_hidden))
current_b2 = np.zeros((1,n_hidden2))
current_w2 = np.random.normal(0, np.sqrt(2.0/(n_hidden+n_hidden2)), (n_hidden,n_hidden2))
current_b3 = np.zeros((1,n_out))
current_w3 = np.random.normal(0, np.sqrt(2.0/(n_hidden2+n_out)), (n_hidden2,n_out))

rmsprop_b1 = np.zeros_like(current_b1)
rmsprop_w1 = np.zeros_like(current_w1)
rmsprop_b2 = np.zeros_like(current_b2)
rmsprop_w2 = np.zeros_like(current_w2)
rmsprop_b3 = np.zeros_like(current_b3)
rmsprop_w3 = np.zeros_like(current_w3)


def forward(b1,w1,b2,w2,b3,w3,inp):
    h = inp.dot(w1) + b1
    h[h<0] = 0
    h2 = h.dot(w2) + b2
    h2[h2<0] = 0
    out = h2.dot(w3) + b3
    return np.exp(out) / (np.sum(np.exp(out)) + 1e-20)

def get_action_current(state):
    return get_action(current_b1, current_w1, current_b2, current_w2, current_b3, current_w3, state)
    
def get_action(b1,w1,b2,w2,b3,w3,state):
    dist = forward(b1,w1,b2,w2,b3,w3,state).reshape(4)
    #return np.argmax(dist)
    return np.random.choice(4, 1, p=dist)[0]

def fitness_current(iters):
    return fitness(current_b1, current_w1, current_b2, current_w2, current_b3, current_w3, iters)

def fitness(b1,w1,b2,w2,b3,w3,iters):
    slider.reset()
    accReward = 0
    state = slider.get_state()
    for _ in range(iters):
        action = get_action(b1,w1,b2,w2,b3,w3,state)
        reward,state = slider.step(action)
        accReward += reward
        
    return accReward

decay = 0.9

def agent_loop(iters):
    slider.reset()
    for _ in range(iters):
        state = slider.get_state()
        action = get_action_current(state)
        _,_ = slider.step(action)
        slider.render()
        time.sleep(0.01)

def fitness2(b1,w1,b2,w2,b3,w3):
    #a = get_action(b1,w1,b2,w2,b3,w3,np.array([1,1,1,1,1,1,1,1]))
    #a = np.random.choice(4, 1, p=forward(b1,w1,b2,w2,b3,w3,np.array([1,1,1,1,1,1,1,1])).reshape(4))[0]
    #b = np.random.choice(4, 1, p=forward(b1,w1,b2,w2,b3,w3,np.array([1,0,1,-5,14,1,0,1])).reshape(4))[0]
    #c = np.random.choice(4, 1, p=forward(b1,w1,b2,w2,b3,w3,np.array([0,1,0,1,0,1,0,1])).reshape(4))[0]
    #d = np.random.choice(4, 1, p=forward(b1,w1,b2,w2,b3,w3,np.array([1,1,4,1,-1,1,11,0])).reshape(4))[0]
    #e = np.random.choice(4, 1, p=forward(b1,w1,b2,w2,b3,w3,np.array([0,1,-3,1,1,5,1,1])).reshape(4))[0]
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

    #return -np.square(dist[3] - 1)

#train(20,0.01,0.01,10,4000)
def train(npop, sigma, alpha, gen, episodelengths):
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
    for i in range(gen):

        #if i % 5 == 0:
        #    print(i, " ", fitness(current_b1, current_w1, current_b2, current_w2, current_b3, current_w3, 2000), " sigma: ", sigma)
            #sigma = sigma*0.99     

        #N_b1 = np.random.normal(0, np.sqrt(2.0/(1+n_hidden)), (npop, 1, n_hidden))
        N_b1 = np.random.randn(npop, 1, n_hidden)/n_hidden
        #N_b1 = np.concatenate((N_b1, -N_b1))
        #N_w1 = np.random.normal(0, np.sqrt(2.0/(n_inputs+n_hidden)), (npop, n_inputs, n_hidden))
        N_w1 = np.random.randn(npop, n_inputs, n_hidden)
        #N_w1 = np.concatenate((N_w1, -N_w1))
        #N_b2 = np.random.normal(0, np.sqrt(2.0/(1+n_hidden2)), (npop, 1, n_hidden2))
        N_b2 = np.random.randn(npop, 1, n_hidden2)/n_hidden2
        #N_b2 = np.concatenate((N_b2, -N_b2))
        #N_w2 = np.random.normal(0, np.sqrt(2.0/(n_hidden+n_hidden2)), (npop, n_hidden, n_hidden2))
        N_w2 = np.random.randn(npop, n_hidden, n_hidden2)
        #N_w2 = np.concatenate((N_w2, -N_w2))
        #N_b3 = np.random.normal(0, np.sqrt(2.0/(1+n_out)), (npop, 1, n_out))
        N_b3 = np.random.randn(npop, 1, n_out)/n_out
        #N_b3 = np.concatenate((N_b3, -N_b3))
        #N_w3 = np.random.normal(0, np.sqrt(2.0/(n_hidden2+n_out)), (npop, n_hidden2, n_out))
        N_w3 = np.random.randn(npop, n_hidden2, n_out)
        #N_w3 = np.concatenate((N_w3, -N_w3))
        
        R = np.zeros(npop)
        
        for j in range(npop):
            b1_try = current_b1 + sigma*N_b1[j]
            w1_try = current_w1 + sigma*N_w1[j]
            b2_try = current_b2 + sigma*N_b2[j]
            w2_try = current_w2 + sigma*N_w2[j]
            b3_try = current_b3 + sigma*N_b3[j]
            w3_try = current_w3 + sigma*N_w3[j]
            R[j] = fitness(b1_try, w1_try, b2_try, w2_try, b3_try, w3_try, episodelengths)
            #R[j] = fitness2(b1_try, w1_try, b2_try, w2_try, b3_try, w3_try) 

        m = np.mean(R)
        if i == 0:
            mr = m
        else:
            mr = 0.9*mr + (0.1)*m

        ph = (episodelengths/1000)
        print (mr/ph, " ", m/ph)

        t = R.argsort()
        ranks = np.empty_like(t)
        ranks[t] = np.arange(len(R))
        A = (ranks - np.mean(ranks)) / (np.std(ranks)*3 + 0.000001)
        #A = (R - np.mean(R)) / (np.std(R) + 0.000001)

        delta_b1 =  1/(npop*sigma) * np.dot(N_b1.T, A).T
        delta_w1 =  1/(npop*sigma) * np.dot(N_w1.T, A).T
        delta_b2 =  1/(npop*sigma) * np.dot(N_b2.T, A).T
        delta_w2 =  1/(npop*sigma) * np.dot(N_w2.T, A).T
        delta_b3 =  1/(npop*sigma) * np.dot(N_b3.T, A).T
        delta_w3 =  1/(npop*sigma) * np.dot(N_w3.T, A).T

        #if i == 0:
        #    rmsprop_b1 = delta_b1**2
        #    rmsprop_21 = delta_w1**2
        #    rmsprop_b2 = delta_b2**2
        #    rmsprop_w2 = delta_w2**2
        #    rmsprop_b3 = delta_b3**2
        #    rmsprop_w3 = delta_w3**2
        #else:
        rmsprop_b1 = decay*rmsprop_b1 + (1-decay)*(delta_b1**2)
        rmsprop_w1 = decay*rmsprop_w1 + (1-decay)*(delta_w1**2)
        rmsprop_b2 = decay*rmsprop_b2 + (1-decay)*(delta_b2**2)
        rmsprop_w2 = decay*rmsprop_w2 + (1-decay)*(delta_w2**2)
        rmsprop_b3 = decay*rmsprop_b3 + (1-decay)*(delta_b3**2)
        rmsprop_w3 = decay*rmsprop_w3 + (1-decay)*(delta_w3**2)

        current_b1 += alpha * delta_b1/(np.sqrt(rmsprop_b1) + 1e-5)
        current_w1 += alpha * delta_w1/(np.sqrt(rmsprop_w1) + 1e-5)
        current_b2 += alpha * delta_b2/(np.sqrt(rmsprop_b2) + 1e-5)
        current_w2 += alpha * delta_w2/(np.sqrt(rmsprop_w2) + 1e-5)
        current_b3 += alpha * delta_b3/(np.sqrt(rmsprop_b3) + 1e-5)
        current_w3 += alpha * delta_w3/(np.sqrt(rmsprop_w3) + 1e-5)

        #current_b1 +=  alpha * delta_b1
        #urrent_w1 +=  alpha * delta_w1
        #current_b2 +=  alpha * delta_b2
        #current_w2 +=  alpha * delta_w2
        #current_b3 +=  alpha * delta_b3
        #current_w3 +=  alpha * delta_w3
