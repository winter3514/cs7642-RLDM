# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 22:49:20 2020

@author: Shuai Jia
"""

import numpy as np
from matplotlib import pyplot as plt

def state_nei(state):
    neis = {'B':['A', 'C'],
            'C':['B', 'D'],
            'D':['C', 'E'],
            'E':['D', 'F'],
            'F':['E', 'G']}
    return neis[state]


def state_to_vec(state):
    vecs = {'B':np.array([1,0,0,0,0]),
            'C':np.array([0,1,0,0,0]),
            'D':np.array([0,0,1,0,0]),
            'E':np.array([0,0,0,1,0]),
            'F':np.array([0,0,0,0,1])}
    return vecs[state]

def generate_sequence():
    init_state = 'D'
    seq = [init_state]
    seq_vec = [state_to_vec(init_state)]
    cur_state = init_state
    reward = 0
    while (cur_state not in ['A', 'G']):
        neis = state_nei(cur_state)
        next_state = np.random.choice(neis)
        seq.append(next_state)
        if next_state not in ['A', 'G']:
            seq_vec.append(state_to_vec(next_state))
        elif next_state == 'A':
            reward = 0
        elif next_state == 'G':
            reward = 1
        cur_state = next_state
    return (seq, seq_vec, reward)

#seq, seq_vec, reward = generate_sequence()
#print(seq)
#print(seq_vec)
#print(reward)
    
def generate_trainingsets(num_training_sets, num_seqs):
    training_sets = []
    reward_sets = []
    for i in range(num_training_sets):
        training_seqs = []
        reward_seqs = []
        for j in range(num_seqs):
            _, seq_vec, reward = generate_sequence()
            training_seqs.append(seq_vec)
            reward_seqs.append(reward)
        training_sets.append(training_seqs)
        reward_sets.append(reward_seqs)
    return (training_sets, reward_sets)

#training_sets, reward_sets = generate_trainingsets(3, 2)
#print(training_sets)
#print(reward_sets)
    
# train weight with one training set (10 training seqs)
def td_training_seqs(training_seqs, reward_seqs, lambd, alpha = 0.01, epsilon = 0.001):
    w = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    #delta_w = np.array([0,0,0,0,0], dtype = np.float16)
    while True:
        w_old = w.copy()
        #print(w_old)# for converge comparasion
        #delta_w_seq = np.array([0,0,0,0,0], dtype = np.float16)
        for i in range(len(training_seqs)):
            #print(i)
            error = np.array([0,0,0,0,0], dtype = np.float16)
            delta_w = np.array([0,0,0,0,0], dtype = np.float16)
            seqs = training_seqs[i]
            rew = reward_seqs[i]
            for t in range(len(seqs)):
                if t != len(seqs) - 1: #not the last state to terminal state, such as B, F
                    cur_state = seqs[t]
                    cur_pred = np.dot(w, cur_state)
                    next_state = seqs[t+1] 
                    next_pred = np.dot(w, next_state)
                    error = lambd*error + cur_state
                    #print(error.dtype)
                    #print(type(next_pred))
                    delta_w = delta_w + alpha*(next_pred - cur_pred)*error
                else: #the last state, B or F,
                    cur_state = seqs[t]
                    cur_pred = np.dot(w, cur_state)
                    #next state is the terminal state, A or G, but we know the next pred from the rewards
                    error = lambd*error + cur_state
                    delta_w = delta_w + alpha*(rew - cur_pred)*error
            #delta_w_seq += delta_w
        #w += delta_w_seq
            w = w + delta_w
            #update w for every seq
            #w += delta_w
        #print(w)
        #print(w_old)
        change = np.linalg.norm(w_old-w)
        if change <= epsilon:
            break
    return w
                    
#training_sets, reward_sets = generate_trainingsets(3, 10)
#w = td_training_seqs(training_sets[0], reward_sets[0], 0.3)
#print(w)
    
def td_training_sets_err(training_sets, reward_sets, lambd, w_true, alpha = 0.01, epsilon = 0.001):
    num_training_sets = len(training_sets)
    err = 0
    for i in range(num_training_sets):
        training_seqs = training_sets[i]
        reward_seqs = reward_sets[i]
        w_td = td_training_seqs(training_seqs, reward_seqs, lambd, alpha)
        err += np.sqrt(np.mean((w_true - w_td)**2))
    avg_err = err / num_training_sets
    return avg_err


def td_training_seqs_update_seq(training_seqs, reward_seqs, lambd, alpha = 0.01, epsilon = 0.001):
    w = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    #delta_w = np.array([0,0,0,0,0], dtype = np.float16)
        #w_old = w.copy()
        #print(w_old)# for converge comparasion
        #delta_w_seq = np.array([0,0,0,0,0], dtype = np.float16)
    for i in range(len(training_seqs)):
        #print(i)
        error = np.array([0,0,0,0,0], dtype = np.float16)
        delta_w = np.array([0,0,0,0,0], dtype = np.float16)
        seqs = training_seqs[i]
        rew = reward_seqs[i]
        for t in range(len(seqs)):
            if t != len(seqs) - 1: #not the last state to terminal state, such as B, F
                cur_state = seqs[t]
                cur_pred = np.dot(w, cur_state)
                next_state = seqs[t+1] 
                next_pred = np.dot(w, next_state)
                error = lambd*error + cur_state
                #print(error.dtype)
                #print(type(next_pred))
                delta_w = delta_w + alpha*(next_pred - cur_pred)*error
            else: #the last state, B or F,
                cur_state = seqs[t]
                cur_pred = np.dot(w, cur_state)
                #next state is the terminal state, A or G, but we know the next pred from the rewards
                error = lambd*error + cur_state
                delta_w = delta_w + alpha*(rew - cur_pred)*error
        #delta_w_seq += delta_w
    #w += delta_w_seq
        w = w + delta_w
            #update w for every seq
            #w += delta_w
        #print(w)
        #print(w_old)
    return w
                    
#training_sets, reward_sets = generate_trainingsets(3, 10)
#w = td_training_seqs(training_sets[0], reward_sets[0], 0.3)
#print(w)
    
def td_training_sets_err_update_seq(training_sets, reward_sets, lambd, w_true, alpha = 0.01, epsilon = 0.001):
    num_training_sets = len(training_sets)
    err = 0
    for i in range(num_training_sets):
        training_seqs = training_sets[i]
        reward_seqs = reward_sets[i]
        w_td = td_training_seqs_update_seq(training_seqs, reward_seqs, lambd, alpha)
        err += np.sqrt(np.mean((w_true - w_td)**2))
    avg_err = err / num_training_sets
    return avg_err
            
#training_sets, reward_sets = generate_trainingsets(3, 10)
#w_true = np.array([1,2,3,4,5])/6
#avg_err = td_training_sets_err(training_sets, reward_sets, 0.3, w_true)
#print(avg_err) 
training_sets, reward_sets = generate_trainingsets(num_training_sets = 100, num_seqs = 10)

def figure3(lambd_values, w_true, num_training_sets = 100, num_seqs = 10):
    avg_err_values = []
    #training_sets, reward_sets = generate_trainingsets(num_training_sets, num_seqs)
    for lambd in lambd_values:
        #training_sets, reward_sets = generate_trainingsets(num_training_sets, num_seqs)
        avg_err = td_training_sets_err(training_sets, reward_sets, lambd, w_true)
        avg_err_values.append(avg_err)
        #print(avg_err)
    plt.plot(lambd_values, avg_err_values, marker = 'o')
    plt.xlabel('Lambda')
    plt.ylabel('Error')
    plt.xlim((min(lambd_values) - 0.05, max(lambd_values) + 0.05))
    plt.ylim((min(avg_err_values) - 0.01, max(avg_err_values) + 0.01))
    plt.margins(x = 0.5, y = 0.15)
    plt.annotate('widow-hoff', xy = (0.75, max(avg_err_values) - 0.01))
    plt.savefig('figure3')
    plt.show()

lambd_values = [0,0.1,0.3,0.5,0.7,0.9,1]
w_true = np.array([0.1666,0.3333,0.5,0.6666,0.8333])
figure3(lambd_values, w_true)

def figure4(lambd_values, alpha_values, w_true, num_training_sets = 100, num_seqs = 10):
    avg_err_lambd_alpha = {}
    #training_sets, reward_sets = generate_trainingsets(num_training_sets, num_seqs)
    
    for lambd in lambd_values:
        avg_err_alpha = []
        for alpha in alpha_values:
            avg_err = td_training_sets_err_update_seq(training_sets, reward_sets, lambd, w_true, alpha)
            avg_err_alpha.append(avg_err)
            #print(avg_err)
        avg_err_lambd_alpha[lambd] = avg_err_alpha
    #plotting
    #fig = plt.figure()
    for lambd in lambd_values:
        plt.plot(alpha_values, avg_err_lambd_alpha[lambd], marker = 'o', label = 'lambda = {}'.format(lambd))
    plt.legend()
    plt.margins(x = 0.5, y = 0.15)
    plt.xlim((min(alpha_values) - 0.05, max(alpha_values) + 0.05))
    plt.ylim((0.05, 0.75))
    plt.xlabel('Alpha')
    plt.ylabel('Error')   
    plt.savefig('figure4')
    plt.show()
    
lambd_values = np.array([0,0.3,0.8,1])
alpha_values = np.array([0.05*i for i in range(13)])
#print(alpha_values)
w_true = np.array([1,2,3,4,5])/6
figure4(lambd_values, alpha_values, w_true)



def figure5(lambd_values, alpha_values, w_true, num_training_sets = 100, num_seqs = 10):
    #avg_err_lambd_alpha = {}
    lambd_alpha = {}
    #training_sets, reward_sets = generate_trainingsets(num_training_sets, num_seqs)
    
    for lambd in lambd_values:
        #avg_err_alpha = []
        min_err = np.inf
        best_alpha = 0
        for alpha in alpha_values:
            avg_err = td_training_sets_err_update_seq(training_sets, reward_sets, lambd, w_true, alpha)
            #avg_err_alpha.append(avg_err)
            if avg_err < min_err:
                best_alpha = alpha
                min_err = avg_err
        #avg_err_lambd_alpha[lambd] = avg_err_alpha
        lambd_alpha[lambd] = best_alpha
    #plotting
    #fig = plt.figure()
    avg_err_lambd_bestalpha = []
    for lambd in lambd_values:
        avg_err = td_training_sets_err_update_seq(training_sets, reward_sets, lambd, w_true, lambd_alpha[lambd])
        avg_err_lambd_bestalpha.append(avg_err)
    plt.plot(lambd_values, avg_err_lambd_bestalpha, marker = 'o')
    plt.margins(x = 0.5, y = 0.15)
    plt.xlim((min(lambd_values) - 0.05, max(lambd_values) + 0.05))
    plt.ylim((min(avg_err_lambd_bestalpha) - 0.01, max(avg_err_lambd_bestalpha) + 0.01))
    plt.xlabel('Lambd')
    plt.ylabel('Error with best alpha')
    plt.annotate('widow-hoff', xy = (0.75, max(avg_err_lambd_bestalpha) - 0.01))
    plt.savefig('figure5')
    plt.show()
    
lambd_values = [i/10 for i in range(11)]
alpha_values = [0.05*i for i in range(13)]
w_true = np.array([1,2,3,4,5])/6
figure5(lambd_values, alpha_values, w_true)
