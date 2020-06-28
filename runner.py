from copy import deepcopy
from master_file import regret_elf, regret_mwu, regret_exp3, regret_wsux, regret_hedge, regret_wsu 
from simulation_parameters import set_params

num_samples_K = 1
num_experts   = 50

def runner(num_experts, sample_id):
    #(num_experts, outcomes, experts_reports, T, num_repetitions) = set_params('reader_forecasts1920.csv',num_experts,sample_id)
    (num_experts, outcomes, experts_reports, T, num_repetitions) = set_params('',num_experts,sample_id)
    
    regret_elf(num_experts, outcomes, experts_reports, T, num_repetitions, sample_id)

    regret_mwu(num_experts, outcomes, experts_reports, T, num_repetitions, sample_id)
    
    regret_wsu(num_experts, outcomes, experts_reports, T, num_repetitions, sample_id)
    
    regret_hedge(num_experts, outcomes, experts_reports, T, num_repetitions, sample_id)
    
    regret_exp3(num_experts, outcomes, experts_reports, T, num_repetitions, sample_id)
    
    regret_wsux(num_experts, outcomes, experts_reports, T, num_repetitions, sample_id)

for i in range(num_samples_K):
    runner(num_experts, i)
