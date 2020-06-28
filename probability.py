'''
    Code for choosing an expert given the probabilities of the experts.
    Gets as input the list of the probabilities for each expert.
    Returns the expert to be drawn.
'''
import random
import numpy as np

# For discrete probability distributions
def draw(probs_lst, gamma, num_experts):
    np.random.seed()
    t           = np.random.uniform(0,1)
    cumulative  = 0.0
    for i in range(num_experts):
        cumulative += (1.0 - gamma)*probs_lst[i] + gamma/num_experts
        if cumulative > t:
            return i
    return (num_experts-1)


def draw_rec(probs_time, gamma, time):
    num_experts = len(probs_time[0])
    if time == 0:
        return [draw(probs_time[0],gamma, num_experts)]
    else: 
        chosen_lst = [draw(probs_time[t], gamma, num_experts) for t in range(time)]
    return (chosen_lst)












