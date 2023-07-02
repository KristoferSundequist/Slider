import lightgbm as lgb
import copy
from memory import Transition
from typing import List
import globals
import numpy as np
import math
import datetime
import utils
#from sklearn.linear_model import LinearRegression

class Policy:
    def __init__(self):
        self.params = {
            'objective': 'regression_l1',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'verbose': -1,
            'min_data_in_leaf': 1
        }

        self._agent = None
        self._lagged_agent = None
    
    def init_agent(self, transitions: List[Transition]):
        state_inputs = np.array([t.state.tolist() + [t.action] for t in transitions])
        self._agent = lgb.train(self.params, lgb.Dataset(state_inputs, np.random.normal(-0.01, 0.01, state_inputs.shape[0])))
        self._lagged_agent = lgb.train(self.params, lgb.Dataset(state_inputs, np.random.normal(-0.01, 0.01, state_inputs.shape[0])))

    def update(self, transitions: List[Transition], lagged: bool = False):
        utils.printtime("1")
        Q_target_input = np.array([t.next_state.tolist() + [a] for t in transitions for a in range(globals.action_space_size)])
        assert(Q_target_input.shape == (len(transitions)*globals.action_space_size, len(transitions[0].state)+1))
        #print(Q_target_input[:10])
        
        utils.printtime("2")
        raw_Q_target_values = self._lagged_agent.predict(Q_target_input)
        assert(raw_Q_target_values.shape == (len(transitions)*globals.action_space_size,))
        #print(raw_Q_target_values[:10])
        
        utils.printtime("3")
        grouped_Q_target_values = np.array(raw_Q_target_values).reshape(len(transitions), globals.action_space_size)
        assert(grouped_Q_target_values.shape == (len(transitions), globals.action_space_size))
        
        indices_of_best = np.expand_dims(grouped_Q_target_values.argmax(1), axis=1)
        
        target_Q_values = np.take_along_axis(grouped_Q_target_values, indices_of_best, axis=1).reshape(-1)
        #print(target_Q_values[:10])
        
        utils.printtime("4")
        target_values = np.array([transitions[i].reward + math.pow(globals.discount_factor, globals.num_steps) * target_Q_values[i] for i in range(len(transitions))])
        assert(target_values.shape == (len(transitions),))
        #print(target_values[:10])

        utils.printtime("5")
        state_inputs = np.array([t.state.tolist() + [t.action] for t in transitions])
        assert(state_inputs.shape == (len(transitions), len(transitions[0].state)+1))
        utils.printtime("6")
        #print(target_values[:200])
        print(f'mean target value: {target_values.mean()}')
        new_agent = lgb.train(self.params, lgb.Dataset(state_inputs, target_values))
        #new_agent = LinearRegression().fit(state_inputs, target_values)
        utils.printtime("7")
        if lagged:
            self._lagged_agent = new_agent
        else:
            self._agent = new_agent
    
    def get_Q_values(self, state):
        input = np.array([state.tolist() + [a] for a in range(globals.action_space_size)])
        assert(input.shape == (globals.action_space_size, len(state)+1))
        result = self._agent.predict(input)
        return result
