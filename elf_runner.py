'''
    File implementing code for ELF-X.
'''

import numpy as np
from regret import *
from probability import draw, draw_rec
import random
import math

'''
    from the internal lottery, pick the expert that has been chosen more times
    if ties, break them uniformly at random
'''
def most_wins(lst, experts):
    freq = [0 for _ in range(experts)]
    for i in lst:
        freq[i] += 1
    
    maxx  = 0
    where = 0
    ties  = 0
    for i in range(experts):
        if freq[i] > maxx:
            maxx  = freq[i]
            where = i
            ties  = 0
        elif freq[i] == maxx:
            ties += 1
    
    if ties:
        tie_index = []
        for i in range(experts):
            if freq[i] == maxx:
                tie_index.append(i)

        np.random.seed()
        where = np.random.choice(tie_index)
    return (where, freq) 


'''
    Computation of WSWM module of ELF-X.
'''
def wswm_compute(wagers, experts_reps, num_experts, outcomes, t):
    new_lst = []
    matrix  = [wagers[i]*(1.0 - (outcomes[t] - experts_reps[i][t])**2) for i in range(num_experts)]
    tot     = sum(matrix)
    for i in range(num_experts): 
        new_lst.append(wagers[i]*(1 + 1 - (outcomes[t] - experts_reps[i][t])**2 - tot))
        
    return new_lst 

def main_elf(num_experts, outcomes, experts_reports, T, rep, sample_id):
    experts_loss_lst = [[] for _ in range(T)]
    elf_loss = [0]*T
    avg_loss = [0]*T
    elf_rep_regr = []
    best_fixed_loss = []
    wins_for_master_file  = [[0 for _ in range(T)] for _ in range(num_experts)]
    
    elf_probs_lst = [[] for _ in range(T)]

    elf_probs_lst[0] = [1.0/num_experts]*num_experts

    for t in range(T):
        print ("Timestep t=%d for ELF."%t)
        exp_chosen_lst = draw_rec(elf_probs_lst, 0.0, t)
        experts_loss_lst[t] = [(outcomes[t] - experts_reports[i][t])**2 for i in range(num_experts)] 

        # at the current timestep, choose the expert with the most wins
        (curr_exp_chosen, wins_lst) = most_wins(exp_chosen_lst, num_experts) 
        elf_loss[t] = experts_loss_lst[t][curr_exp_chosen]
        wins_for_master_file[curr_exp_chosen][t] += 1
        # update the elf probs lst
        wagers = [1.0/num_experts]*num_experts
        new_probs_lst = [a for a in wswm_compute(wagers, experts_reports, num_experts, outcomes, t)]
        elf_probs_lst[t] = new_probs_lst

        # regret computations
        (regr_best, best_fixed) = regret(experts_loss_lst, num_experts, elf_loss, t)

        elf_rep_regr.append(regr_best)
        best_fixed_loss.append(best_fixed)

    return (sample_id, num_experts, elf_rep_regr, [sum(elf_loss[:t+1]) for t in range(T)], wins_for_master_file, best_fixed_loss)
        

