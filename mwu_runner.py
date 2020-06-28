import numpy as np
from probability import *
from regret import *


def main_mwu(num_experts, outcomes, experts_reports, T, rep, sample_id):
    mwu_weights = [1.0]*num_experts
    mwu_probs   = [1.0/num_experts]*num_experts

    experts_loss_lst = [[0]*num_experts for _ in range(T)]
    mwu_loss     = [0]*T
    avg_loss     = [0]*T
    mwu_weighted_loss = [0]*T    
    uniform_fixed_loss = [0]*T
    mwu_rep_regr = []


    eta = np.sqrt(1.0*np.log(num_experts)/(1.0*T))
    for t in range(T): 
        print ("Timestep t=%d for MWU"%t)
        exp_chosen = draw(mwu_probs, 0, num_experts) 
        experts_loss_lst[t] = [(outcomes[t] - experts_reports[i][t])**2 for i in range(num_experts)]   
        mwu_loss[t] = experts_loss_lst[t][exp_chosen]        
        # average report for this round
        avg_rep     = 1.0*sum([experts_reports[i][t] for i in range(num_experts)])/num_experts
        # loss of <mwu weighted avg rep> - loss of <simple avg rep>
        uniform_fixed_loss[t] = (outcomes[t] - avg_rep)**2
        mwu_weighted_rep = 1.0*sum([mwu_probs[i]*experts_reports[i][t] for i in range(num_experts)])
        mwu_weighted_loss[t] = (outcomes[t] - mwu_weighted_rep)**2

        # weight update
        temp = [mwu_weights[i]*(1.0 - eta*experts_loss_lst[t][i]) for i in range(num_experts)]
        mwu_weights = temp

        # probs update
        mwu_probs = [1.0*mwu_weights[i]/sum(mwu_weights) for i in range(num_experts)]

        (regr_best, bf) = regret(experts_loss_lst, num_experts, mwu_loss, t)
        
        mwu_rep_regr.append(regr_best)    

    return (sample_id, num_experts, mwu_rep_regr, [sum(mwu_loss[:t+1]) for t in range(T)], [sum(mwu_weighted_loss[:t+1]) for t in range(T)], [sum(uniform_fixed_loss[:t+1]) for t in range(T)])


