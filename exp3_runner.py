import sys
import math
import numpy as np
from probability import *
from regret import *


def main_exp3(num_experts, outcomes, experts_reports, T, rep, sample_id):
    exp3_weights = [1.0]*num_experts
    exp3_probs   = [1.0/num_experts]*num_experts

    experts_loss_lst = [[0]*num_experts for _ in range(T)]
    est_loss     = [[0]*num_experts for _ in range(T)]
    avg_loss     = [0]*T
    exp3_loss     = [0]*T
    exp3_rep_regr = []


    eta = np.sqrt(2*np.log(num_experts)/(num_experts*T))
    for t in range(T): 
        print ("Timestep t=%d for EXP3"%t)
        exp_chosen = draw(exp3_probs, 0, num_experts) 
        experts_loss_lst[t] = [(outcomes[t] - experts_reports[i][t])**2 for i in range(num_experts)]   
        # unbiased estimator
        est_loss[t][exp_chosen] = 1.0*experts_loss_lst[t][exp_chosen]/exp3_probs[exp_chosen]        
        exp3_loss[t] = experts_loss_lst[t][exp_chosen]

        # weight update according to the estimated losses
        temp  = [exp3_weights[i]*np.exp(-eta*est_loss[t][i]) for i in range(num_experts)]
        exp3_weights = temp

        # probs update
        exp3_probs = [1.0*exp3_weights[i]/sum(exp3_weights) for i in range(num_experts)]

        (regr_best, bf) = regret(experts_loss_lst, num_experts, exp3_loss, t)
        
        exp3_rep_regr.append(regr_best)    

    return (sample_id, num_experts, exp3_rep_regr, [sum(exp3_loss[:t+1]) for t in range(T)])


