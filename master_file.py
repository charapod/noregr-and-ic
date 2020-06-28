'''
    File collecting the results for
    different randomness instantiations of 
    algos. Writes the results back to .txt 
    files.
'''

import numpy as np
import random
import math
from copy import deepcopy
from elf_runner import *
from mwu_runner import *
from wsux_runner import *
from wsu_runner import *
from exp3_runner import *
from hedge_runner import *
from multiprocessing import Pool

elf_regr = [[[] for _ in range(10)] for _ in range(305)]
elf_loss = [[[] for _ in range(10)] for _ in range(305)]
elf_weighted_loss = [[[] for _ in range(10)] for _ in range(305)]
mwu_regr  = [[[] for _ in range(10)] for _ in range(305)]
mwu_loss  = [[[] for _ in range(10)] for _ in range(305)]
mwu_weighted_loss   = [[[] for _ in range(10)] for _ in range(305)]
best_fixed_loss     = [[[] for _ in range(10)] for _ in range(305)]
uniform_fixed_loss  = [[[] for _ in range(10)] for _ in range(305)]
wins_lst  = [[[] for _ in range(10)] for _ in range(305)]


wsux_regr = [[[] for _ in range(10)] for _ in range(305)]
wsux_loss = [[[] for _ in range(10)] for _ in range(305)]
exp3_regr  = [[[] for _ in range(10)] for _ in range(305)]
exp3_loss  = [[[] for _ in range(10)] for _ in range(305)]

wswm_regr = [[[] for _ in range(10)] for _ in range(505)]
wswm_loss = [[[] for _ in range(10)] for _ in range(505)]
wswm_weighted_loss = [[[] for _ in range(10)] for _ in range(505)]

hedge_regr  = [[[] for _ in range(10)] for _ in range(505)]
hedge_loss  = [[[] for _ in range(10)] for _ in range(505)]
hedge_weighted_loss   = [[[] for _ in range(10)] for _ in range(505)]


def log_elf_results(res):
    sid = res[0]
    K   = res[1]
    elf_regr[K][sid].append(res[2])
    elf_loss[K][sid].append(res[3])
    wins_lst[K][sid].append(res[4])
    best_fixed_loss[K][sid].append(res[5])

def log_mwu_results(res):
    sid = res[0]
    K   = res[1]
    mwu_regr[K][sid].append(res[2])
    mwu_loss[K][sid].append(res[3])
    mwu_weighted_loss[K][sid].append(res[4])
    uniform_fixed_loss[K][sid].append(res[5])

def log_wsux_results(res):
    sid = res[0]
    K   = res[1]
    wsux_regr[K][sid].append(res[2])
    wsux_loss[K][sid].append(res[3])
    best_fixed_loss[K][sid].append(res[4])

def log_exp3_results(res):
    sid = res[0]
    K   = res[1]
    exp3_regr[K][sid].append(res[2])
    exp3_loss[K][sid].append(res[3])
    
def log_wswm_results(res):
    sid = res[0]
    K   = res[1]
    wswm_regr[K][sid].append(res[2])
    wswm_loss[K][sid].append(res[3])
    wswm_weighted_loss[K][sid].append(res[4])
    best_fixed_loss[K][sid].append(res[5])

def log_hedge_results(res):
    sid = res[0]
    K   = res[1]
    hedge_regr[K][sid].append(res[2])
    hedge_loss[K][sid].append(res[3])
    hedge_weighted_loss[K][sid].append(res[4])


def regret_elf(num_experts, outcomes, experts_reports, T, num_repetitions,sample_id):
    f1 = "simulation_results/%dexperts/elf_regrets_%dexperts-%d.txt"%(num_experts, num_experts, sample_id)
    f2 = "simulation_results/%dexperts/elf_losses_%dexperts-%d.txt"%(num_experts, num_experts, sample_id)
    f3 = "simulation_results/%dexperts/elf_weighted_losses_%dexperts-%d.txt"%(num_experts, num_experts, sample_id)
    f4 = "simulation_results/%dexperts/best_fixed_losses_%dexperts-%d.txt"%(num_experts, num_experts, sample_id)

    elf_regrets = open(f1, "w")
    elf_losses  = open(f2, "w")
    elf_weighted_losses = open(f3, "w")
    best_fixed_losses   = open(f4, "w")

    pool = Pool(processes = num_repetitions)
    results = [pool.apply_async(main_elf, args = (num_experts, outcomes, experts_reports, T, rep,sample_id), callback = log_elf_results) for rep in range(0,num_repetitions)]

    pool.close()
    pool.join()

    # count number of wins according to list that you got from
    # the runner program
    wins_per_expert = [[0 for _ in range(T)] for _ in range(num_experts)]
    for k in range(num_experts):
        for rep in range(num_repetitions):
            for t in range(T):
                wins_per_expert[k][t] += wins_lst[num_experts][sample_id][rep][k][t]


    # compute what probabilities these wins correspond to
    expert_probs = [[] for _ in range(num_experts)]
    for k in range(0,num_experts):
        temp = [1.0*wins_per_expert[k][t]/num_repetitions for t in range(T)]
        expert_probs[k] = temp


    elf_weighted_loss = [0]*T
    for t in range(T):
        elf_weighted_rep = sum([expert_probs[i][t]*experts_reports[i][t] for i in range(num_experts)])
        loss = (outcomes[t] - elf_weighted_rep)**2 
        if t == 0:
            elf_weighted_loss[0] = loss
        else:
            elf_weighted_loss[t] = elf_weighted_loss[t-1] + loss

    for r in range(0,num_repetitions):
        s = ""
        for t in range(0,T):
            s += ("%.5f "%elf_regr[num_experts][sample_id][r][t])
        s += "\n"
        elf_regrets.write(s) 
    for r in range(0,num_repetitions):
        s = ""
        for t in range(0,T):
            s += ("%.5f "%elf_loss[num_experts][sample_id][r][t])
        s += "\n"
        elf_losses.write(s) 

    for r in range(0,num_repetitions):
        s = ""
        for t in range(0,T):
            s += ("%.5f "%elf_weighted_loss[t])
        s += "\n"
        elf_weighted_losses.write(s) 

    for r in range(0,num_repetitions):
        s = ""
        for t in range(0,T):
            s += ("%.5f "%best_fixed_loss[num_experts][sample_id][r][t])
        s += "\n"
        best_fixed_losses.write(s) 

    elf_regrets.close()
    elf_losses.close()
    elf_weighted_losses.close()
    best_fixed_losses.close()
    
    return 


def regret_mwu(num_experts, outcomes, experts_reports, T, num_repetitions, sample_id):
    f1 = "simulation_results/%dexperts/mwu_regrets_%dexperts-%d.txt"%(num_experts, num_experts, sample_id)
    f2 = "simulation_results/%dexperts/mwu_losses_%dexperts-%d.txt"%(num_experts, num_experts, sample_id)
    f3 = "simulation_results/%dexperts/mwu_weighted_losses_%dexperts-%d.txt"%(num_experts, num_experts, sample_id)
    f4 = "simulation_results/%dexperts/uniform_fixed_losses_%dexperts-%d.txt"%(num_experts, num_experts, sample_id)
    mwu_regrets = open(f1, "w")
    mwu_losses  = open(f2, "w")
    mwu_weighted_losses  = open(f3, "w")
    uniform_fixed_losses = open(f4, "w")
        

    pool = Pool(processes = num_repetitions)
    results = [pool.apply_async(main_mwu, args = (num_experts, outcomes, experts_reports, T, rep, sample_id), callback = log_mwu_results) for rep in range(0,num_repetitions)]

    pool.close()
    pool.join()

    for r in range(0,num_repetitions):
        s = ""
        for t in range(0,T):
            s += ("%.5f "%mwu_regr[num_experts][sample_id][r][t])
        s += "\n"
        mwu_regrets.write(s) 
    for r in range(0,num_repetitions):
        s = ""
        for t in range(0,T):
            s += ("%.5f "%mwu_loss[num_experts][sample_id][r][t])
        s += "\n"
        mwu_losses.write(s) 
    for r in range(0,num_repetitions):
        s = ""
        for t in range(0,T):
            s += ("%.5f "%mwu_weighted_loss[num_experts][sample_id][r][t])
        s += "\n"
        mwu_weighted_losses.write(s) 
    for r in range(0,num_repetitions):
        s = ""
        for t in range(0,T):
            s += ("%.5f "%uniform_fixed_loss[num_experts][sample_id][r][t])
        s += "\n"
        uniform_fixed_losses.write(s) 

    mwu_regrets.close()
    mwu_losses.close()
    mwu_weighted_losses.close()
    uniform_fixed_losses.close()

    return 

def regret_wsux(num_experts, outcomes, experts_reports, T, num_repetitions, sample_id):
    f1 = "simulation_results/%dexperts/wsux_regrets_%dexperts-%d.txt"%(num_experts, num_experts, sample_id)
    f2 = "simulation_results/%dexperts/wsux_losses_%dexperts-%d.txt"%(num_experts, num_experts, sample_id)
    f4 = "simulation_results/%dexperts/best_fixed_losses_%dexperts-%d.txt"%(num_experts, num_experts, sample_id)

    wsux_regrets = open(f1, "w")
    wsux_losses  = open(f2, "w")
    best_fixed_losses     = open(f4, "w")

    pool = Pool(processes = num_repetitions)
    results = [pool.apply_async(main_wsux, args = (num_experts, outcomes, experts_reports, T, rep, sample_id), callback = log_wsux_results) for rep in range(0,num_repetitions)]

    pool.close()
    pool.join()

    for r in range(0,num_repetitions):
        s = ""
        for t in range(0,T):
            s += ("%.5f "%wsux_regr[num_experts][sample_id][r][t])
        s += "\n"
        wsux_regrets.write(s) 
    for r in range(0,num_repetitions):
        s = ""
        for t in range(0,T):
            s += ("%.5f "%wsux_loss[num_experts][sample_id][r][t])
        s += "\n"
        wsux_losses.write(s) 
    for r in range(0,num_repetitions):
        s = ""
        for t in range(0,T):
            s += ("%.5f "%best_fixed_loss[num_experts][sample_id][r][t])
        s += "\n"
        best_fixed_losses.write(s) 

    wsux_regrets.close()
    wsux_losses.close()
    best_fixed_losses.close()
    return 


def regret_exp3(num_experts, outcomes, experts_reports, T, num_repetitions, sample_id):
    f1 = "simulation_results/%dexperts/exp3_regrets_%dexperts-%d.txt"%(num_experts, num_experts, sample_id)
    f2 = "simulation_results/%dexperts/exp3_losses_%dexperts-%d.txt"%(num_experts, num_experts, sample_id)
    exp3_regrets = open(f1, "w")
    exp3_losses  = open(f2, "w")
        

    pool = Pool(processes = num_repetitions)
    results = [pool.apply_async(main_exp3, args = (num_experts, outcomes, experts_reports, T, rep, sample_id), callback = log_exp3_results) for rep in range(0,num_repetitions)]

    pool.close()
    pool.join()

    for r in range(0,num_repetitions):
        s = ""
        for t in range(0,T):
            s += ("%.5f "%exp3_regr[num_experts][sample_id][r][t])
        s += "\n"
        exp3_regrets.write(s) 
    for r in range(0,num_repetitions):
        s = ""
        for t in range(0,T):
            s += ("%.5f "%exp3_loss[num_experts][sample_id][r][t])
        s += "\n"
        exp3_losses.write(s) 

    exp3_regrets.close()
    exp3_losses.close()

    return 

def regret_wsu(num_experts, outcomes, experts_reports, T, num_repetitions, sample_id):
    f1 = "simulation_results/%dexperts/wswm_regrets_%dexperts-%d.txt"%(num_experts, num_experts, sample_id)
    f2 = "simulation_results/%dexperts/wswm_losses_%dexperts-%d.txt"%(num_experts, num_experts, sample_id)
    f3 = "simulation_results/%dexperts/wswm_weighted_losses_%dexperts-%d.txt"%(num_experts, num_experts, sample_id)
    f4 = "simulation_results/%dexperts/best_fixed_losses_%dexperts-%d.txt"%(num_experts, num_experts, sample_id)

    wswm_regrets = open(f1, "w")
    wswm_losses  = open(f2, "w")
    wswm_weighted_losses  = open(f3, "w")
    best_fixed_losses     = open(f4, "w")

    #wswm_regr is now a num_repetitionsxT matrix
    pool = Pool(processes = num_repetitions)
    results = [pool.apply_async(main_wswm, args = (num_experts, outcomes, experts_reports, T, rep, sample_id), callback = log_wswm_results) for rep in range(0,num_repetitions)]

    pool.close()
    pool.join()

    for r in range(0,num_repetitions):
        s = ""
        for t in range(0,T):
            s += ("%.5f "%wswm_regr[num_experts][sample_id][r][t])
        s += "\n"
        wswm_regrets.write(s) 
    for r in range(0,num_repetitions):
        s = ""
        for t in range(0,T):
            s += ("%.5f "%wswm_loss[num_experts][sample_id][r][t])
        s += "\n"
        wswm_losses.write(s) 
    for r in range(0,num_repetitions):
        s = ""
        for t in range(0,T):
            s += ("%.5f "%wswm_weighted_loss[num_experts][sample_id][r][t])
        s += "\n"
        wswm_weighted_losses.write(s) 
    for r in range(0,num_repetitions):
        s = ""
        for t in range(0,T):
            s += ("%.5f "%best_fixed_loss[num_experts][sample_id][r][t])
        s += "\n"
        best_fixed_losses.write(s) 

    wswm_regrets.close()
    wswm_losses.close()
    wswm_weighted_losses.close()
    best_fixed_losses.close()
    return 

def regret_hedge(num_experts, outcomes, experts_reports, T, num_repetitions, sample_id):
    f1 = "simulation_results/%dexperts/hedge_regrets_%dexperts-%d.txt"%(num_experts, num_experts, sample_id)
    f2 = "simulation_results/%dexperts/hedge_losses_%dexperts-%d.txt"%(num_experts, num_experts, sample_id)
    f3 = "simulation_results/%dexperts/hedge_weighted_losses_%dexperts-%d.txt"%(num_experts, num_experts, sample_id)
    hedge_regrets = open(f1, "w")
    hedge_losses  = open(f2, "w")
    hedge_weighted_losses  = open(f3, "w")
        

    #mwu_regr is a num_repetitionsxT matrix
    pool = Pool(processes = num_repetitions)
    results = [pool.apply_async(main_hedge, args = (num_experts, outcomes, experts_reports, T, rep, sample_id), callback = log_hedge_results) for rep in range(0,num_repetitions)]

    pool.close()
    pool.join()

    for r in range(0,num_repetitions):
        s = ""
        for t in range(0,T):
            s += ("%.5f "%hedge_regr[num_experts][sample_id][r][t])
        s += "\n"
        hedge_regrets.write(s) 
    for r in range(0,num_repetitions):
        s = ""
        for t in range(0,T):
            s += ("%.5f "%hedge_loss[num_experts][sample_id][r][t])
        s += "\n"
        hedge_losses.write(s) 
    for r in range(0,num_repetitions):
        s = ""
        for t in range(0,T):
            s += ("%.5f "%hedge_weighted_loss[num_experts][sample_id][r][t])
        s += "\n"
        hedge_weighted_losses.write(s) 

    hedge_regrets.close()
    hedge_losses.close()
    hedge_weighted_losses.close()

    return 
