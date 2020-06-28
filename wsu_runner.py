import numpy as np
from probability import *
from regret import *
from copy import deepcopy

def wswm_compute(wagers, experts_reps, num_experts, outcomes, t):
    new_lst = []
    matrix  = [wagers[i]*(1.0 - (outcomes[t] - experts_reps[i][t])**2) for i in range(num_experts)]
    tot     = sum(matrix)
    for i in range(num_experts): 
        new_lst.append(wagers[i]*(1 + 1 - (outcomes[t] - experts_reps[i][t])**2 - tot))
        
    return new_lst 

def main_wswm(num_experts, outcomes, experts_reports, T, rep, sample_id):
    wswm_probs   = [1.0/num_experts]*num_experts
    experts_loss_lst = [[] for _ in range(T)]
    wswm_loss    = [0]*T
    wswm_weighted_loss = [0]*T
    wswm_rep_regr = []
    best_fixed_loss  = []

    eta = np.sqrt(1.0*np.log(num_experts)/(1.0*T))
    for t in range(T):
        print ("Timestep t=%d for WSWM"%t)
        exp_chosen = draw(wswm_probs, 0, num_experts)
        
        experts_loss_lst[t] = [(outcomes[t] - experts_reports[i][t])**2 for i in range(num_experts)] 
        wswm_loss[t] = experts_loss_lst[t][exp_chosen]        
        # loss of <wswm weighted avg rep> - loss of <simple avg rep>
        wswm_weighted_rep = 1.0*sum([wswm_probs[i]*experts_reports[i][t] for i in range(num_experts)])
        wswm_weighted_loss[t] = (outcomes[t] - wswm_weighted_rep)**2

        # probs update through wswm
        cpy  = deepcopy(wswm_probs)
        temp = wswm_compute(wswm_probs, experts_reports, num_experts, outcomes, t)
        wswm_probs = [eta*temp[i] + (1.0 - eta)*cpy[i] for i in range(num_experts)]

        (regr_best, best_fixed) = regret(experts_loss_lst, num_experts, wswm_loss, t)

        wswm_rep_regr.append(regr_best)
        best_fixed_loss.append(best_fixed)

    return (sample_id, num_experts, wswm_rep_regr, [sum(wswm_loss[:t+1]) for t in range(T)], [sum(wswm_weighted_loss[:t+1]) for t in range(T)], best_fixed_loss)

