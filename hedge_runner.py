import numpy as np
from probability import *
from regret import *


def main_hedge(num_experts, outcomes, experts_reports, T, rep, sample_id):
    hedge_weights = [1.0]*num_experts
    hedge_probs   = [1.0/num_experts]*num_experts

    experts_loss_lst = [[0]*num_experts for _ in range(T)]
    hedge_loss     = [0]*T
    avg_loss     = [0]*T
    hedge_weighted_loss = [0]*T    
    hedge_rep_regr = []


    eta = 1.0/4.0
    for t in range(T): 
        print ("Timestep t=%d for Hedge"%t)
        exp_chosen = draw(hedge_probs, 0, num_experts) 
        experts_loss_lst[t] = [(outcomes[t] - experts_reports[i][t])**2 for i in range(num_experts)]   
        hedge_loss[t] = experts_loss_lst[t][exp_chosen]        
        # average report for this round
        avg_rep     = 1.0*sum([experts_reports[i][t] for i in range(num_experts)])/num_experts
        hedge_weighted_rep = 1.0*sum([hedge_probs[i]*experts_reports[i][t] for i in range(num_experts)])
        hedge_weighted_loss[t] = (outcomes[t] - hedge_weighted_rep)**2

        # weight update
        temp = [hedge_weights[i]*(np.exp(- eta*experts_loss_lst[t][i])) for i in range(num_experts)]
        hedge_weights = temp

        # probs update
        hedge_probs = [1.0*hedge_weights[i]/sum(hedge_weights) for i in range(num_experts)]

        (regr_best, bf) = regret(experts_loss_lst, num_experts, hedge_loss, t)
        
        hedge_rep_regr.append(regr_best)    

    return (sample_id, num_experts, hedge_rep_regr, [sum(hedge_loss[:t+1]) for t in range(T)], [sum(hedge_weighted_loss[:t+1]) for t in range(T)])


