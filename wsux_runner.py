import numpy as np
from probability import *
from regret import *
from copy import deepcopy
import math

def wswm_compute(wagers, experts_reps, num_experts, outcomes, t):
    new_lst = []
    matrix  = [wagers[i]*(1.0 - (outcomes[t] - experts_reps[i][t])**2) for i in range(num_experts)]
    tot     = sum(matrix)
    for i in range(num_experts): 
        new_lst.append(wagers[i]*(1 + 1 - (outcomes[t] - experts_reps[i][t])**2 - tot))
        
    return new_lst # updated probs

def main_wsux(num_experts, outcomes, experts_reports, T, rep, sample_id):
    wsux_probs   = [1.0/num_experts]*num_experts
    experts_loss_lst = [[0]*num_experts for _ in range(T)]
    wsux_loss    = [0]*T
    wsux_weighted_loss = [0]*T
    wsux_rep_regr = []
    best_fixed_loss  = []
    est_loss = [[0]*num_experts for _ in range(T)]

    eta = (1.0*np.log(num_experts)/(4*np.sqrt(num_experts)*T))**(2.0/3.0) 
    gamma = np.sqrt(1.0*eta*num_experts)

    for t in range(T):
        print ("Timestep t=%d for WSU-UX"%t)
        exp_chosen = draw(wsux_probs, gamma, num_experts)
        
        experts_loss_lst[t] = [(outcomes[t] - experts_reports[i][t])**2 for i in range(num_experts)] 
        wsux_loss[t] = experts_loss_lst[t][exp_chosen]        
        est_loss[t][exp_chosen] = 1.0*experts_loss_lst[t][exp_chosen]/wsux_probs[exp_chosen] 

        # probs update through wswm
        cpy  = deepcopy(wsux_probs)
        temp = wswm_compute(wsux_probs, experts_reports, num_experts, outcomes, t)
        wsux_probs = [eta*temp[i] + (1.0 - eta)*cpy[i] for i in range(num_experts)]

        (regr_best, best_fixed) = regret(experts_loss_lst, num_experts, wsux_loss, t)

        wsux_rep_regr.append(regr_best)
        best_fixed_loss.append(best_fixed)

    return (sample_id, num_experts, wsux_rep_regr, [sum(wsux_loss[:t+1]) for t in range(T)], [sum(wsux_weighted_loss[:t+1]) for t in range(T)], best_fixed_loss)

