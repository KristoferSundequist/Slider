from multiprocessing import Pool
import numpy as np
import math
import slider
import time
	
np.random.seed(0)

# RMSprop decay
decay = 0.9

# decay factor
weight_decay=0.01

# Network params
n_inputs = 8
n_hidden = 200
n_hidden2 = 200
n_out = 4;

class Nn:
    def __init__(self):
        self.b1 = np.zeros((1,n_hidden))
        self.w1 = np.zeros((n_inputs,n_hidden))
        self.b2 = np.zeros((1,n_hidden2))
        self.w2 = np.zeros((n_hidden,n_hidden2))
        self.b3 = np.zeros((1,n_out))
        self.w3 = np.zeros((n_hidden2,n_out))        
        
    def init_weights(self):
        self.b1 = np.random.normal(0, 0.5, (1,n_hidden))
        self.w1 = np.random.normal(0, np.sqrt(2.0/(n_inputs+n_hidden)), (n_inputs,n_hidden))
        self.b2 = np.random.normal(0,0.5,(1,n_hidden2))
        self.w2 = np.random.normal(0, np.sqrt(2.0/(n_hidden+n_hidden2)), (n_hidden,n_hidden2))
        self.b3 = np.random.normal(0,0.5,(1,n_out))
        self.w3 = np.random.normal(0, np.sqrt(2.0/(n_hidden2+n_out)), (n_hidden2,n_out))

    def add(self, other):
        self.b1 += other.b1
        self.w1 += other.w1
        self.b2 += other.b2
        self.w2 += other.w2
        self.b3 += other.b3
        self.w3 += other.w3

    def subtract(self, other):
        self.b1 -= other.b1
        self.w1 -= other.w1
        self.b2 -= other.b2
        self.w2 -= other.w2
        self.b3 -= other.b3
        self.w3 -= other.w3

    def multiply_uniform(self, lower, upper):
        self.b1 *= np.random.uniform(lower,upper,(1,n_hidden))
        self.w1 *= np.random.uniform(lower,upper,(n_inputs,n_hidden))
        self.b2 *= np.random.uniform(lower,upper,(1,n_hidden2))
        self.w2 *= np.random.uniform(lower,upper,(n_hidden,n_hidden2))
        self.b3 *= np.random.uniform(lower,upper,(1,n_out))
        self.w3 *= np.random.uniform(lower,upper,(n_hidden2,n_out))

    def clip(self, lower, upper):
        self.b1 = np.clip(self.b1,lower,upper)
        self.w1 = np.clip(self.w1,lower,upper)
        self.b2 = np.clip(self.b2,lower,upper)
        self.w2 = np.clip(self.w2,lower,upper)
        self.b3 = np.clip(self.b3,lower,upper)
        self.w3 = np.clip(self.w3,lower,upper)
    
    def multiply_scalar(self, scalar):
        self.b1 *= scalar
        self.w1 *= scalar
        self.b2 *= scalar
        self.w2 *= scalar
        self.b3 *= scalar
        self.w3 *= scalar

    def copy(self, other):
        self.b1 = np.copy(other.b1)
        self.w1 = np.copy(other.w1)
        self.b2 = np.copy(other.b2)
        self.w2 = np.copy(other.w2)
        self.b3 = np.copy(other.b3)
        self.w3 = np.copy(other.w3)
        
    def forward(self, inp):
        h = inp.dot(self.w1) + self.b1
        #h[h<0] = 0
        h = 1/(1+np.exp(-h))
        h2 = h.dot(self.w2) + self.b2
        #h2[h2<0] = 0
        h2 = 1/(1+np.exp(-h2))
        out = h2.dot(self.w3) + self.b3
        a = np.exp(out)
        b = np.sum(a) + 1e-5
        #np.asarray_chkfinite(a)
        #np.asarray_chkfinite(b)
        return a/b

    # Samples an action using weights and given state
    def get_action(self,inp):
        dist = self.forward(inp).reshape(n_out)
        return np.argmax(dist)

    
class Particle:
    def __init__(self):
        self.weights = Nn()
        self.weights.init_weights()
        
        self.velocity = Nn()

        self.best_score = 0
        self.best = Nn()

    def update(self, phi_cognitive, phi_social, gbest):
        #cognivite
        cognitive = Nn()
        cognitive.copy(self.best)
        cognitive.subtract(self.weights)
        cognitive.multiply_uniform(0,phi_cognitive)

        #social
        social = Nn()
        social.copy(gbest)
        social.subtract(self.weights)
        social.multiply_uniform(0,phi_social)

        #merge
        cognitive.add(social)
        #cognitive.multiply_scalar(0.7) #constriction

        #update velocity and weights
        self.velocity.add(cognitive)
        
        self.velocity.clip(-3,3)
        
        self.weights.add(self.velocity)
        #self.weights.clip(-100,100)
        #self.weights.multiply_scalar(0.5)


    def fitness4(self):
        a = self.weights.get_action(np.array([   1,   1,   1, 0.3,  1, 0.1, 1,  1]))
        b = self.weights.get_action(np.array([   1,   0, 0.1,  -1,  1,   1, 0,  1]))
        c = self.weights.get_action(np.array([   0,   1,   0,   1,  0,   1, 0,  1]))
        d = self.weights.get_action(np.array([   1,   1,   1,   1, -1,   1, 1,  0]))
        e = self.weights.get_action(np.array([   0,   1,  -1,   1,0.5, 0.6, 1,  1]))
        f = self.weights.get_action(np.array([   0,   1,-0.5, 0.7,  1,   1, 1, -1]))
        g = self.weights.get_action(np.array([   1,   0,   1,   0, -1,   0, 0,  1]))
        h = self.weights.get_action(np.array([  -1,  -1,   0,  -1,  0,   1, 0,  1]))
        i = self.weights.get_action(np.array([   1,   1,   1,-0.5,  1,-0.5, 1,  0]))
        j = self.weights.get_action(np.array([   1,   1,  -1,   0,  0,   1,-1, -1]))

        fit = int(a==3) + int(b==1) + int(c==3) + int(d==2) + int(e==3) + int(f==3) + int(g==2) + int(h==0) + int(i==1) + int(j==0)
        return fit

    def evaluate(self):
        f = self.fitness4()
        if f > self.best_score:
            self.best_score = f
            self.best.copy(self.weights)
        return f

class Swarm:
    def __init__(self, n_particles):
        self.particles = []
        for p in range(n_particles):
            self.particles.append(Particle())

        self.best_score = 0
        self.best = Nn()
        
    def update(self):
        qwe = 0
        for p in self.particles:
            #print(qwe)
            qwe += 1
            f = p.evaluate()
            if f > self.best_score:
                self.best_score = f
                self.best.copy(p.weights)
            p.update(2,2,self.best)

    def train(self, iters):
        for i in range(iters):
            print(self.best_score)
            self.update()

    def eval_best(self):
        p = Particle()
        p.weights = self.best
        return p.evaluate()

            
        
        
        
    

# Eval some weights for iters steps (episode size)
def fitness(b1,w1,b2,w2,b3,w3,iters):
    slider.reset()
    accReward = 0
    state = slider.get_state()
    for _ in range(iters):
        action = get_action(b1,w1,b2,w2,b3,w3,state)
        reward,state = slider.step(action)
        accReward += reward
        
    return accReward# - l2loss(b1,w1,b2,w2,b3,w3)

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
