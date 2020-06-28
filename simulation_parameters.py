''' 
    Specify params of simulation to be run.
'''

import numpy as np
import random
import math
import csv

def set_params(dataset,num_experts,sample_id):
    num_repetitions = 50
    if (dataset):
        input_file = csv.DictReader(open(dataset))

        data = []
        for row in input_file: 
            data.append(row)

        forecasters_sort = sorted(data, key = lambda i: i['user_id'])
        teams_sort       = sorted(data, key = lambda i: i['team1'])

        diff_forecasters = []
        prev             = '' 
        for row in forecasters_sort: 
            if row['user_id'] != prev:
                diff_forecasters.append(row['user_id']) # names of diff forecasters
                prev = row['user_id']

        print ("Num of diff forecasters is: %d"%len(diff_forecasters))

        num_teams = 0
        prev      = ''
        for row in teams_sort:
            if row['team1'] != prev:
                prev = row['team1']
                num_teams += 1

        print ("Num of diff teams is %d"%num_teams)

        num_matches = 0
        against     = []

        for row in teams_sort:
            match = {'week' : row['week'], 'team1': row['team1'], 'team2' : row['team2']}
            if match not in against:
                against.append(match)
                num_matches += 1

        print ("Num of diff nfl games is: %d"%num_matches)

        num_preds = [0]*len(diff_forecasters)
        prev      = ''

        for row in forecasters_sort:
            if row['user_id']!=prev:
                index = int(row['user_id'])
                num_preds[index] += 1
                prev = row['user_id']
            else:
                num_preds[int(prev)] += 1

        print ("Number of forecasters reporting on all nfl games.")
        num_preds.sort()
        print sum([1 if num_preds[i] == 267 else 0 for i in range(len(diff_forecasters))])

        experts = []
        prev    = 0
        for i in range(len(num_preds)):
            if num_preds[i] == 267:
                experts.append(i)

        for i in range(len(experts)):
            if i >= 1 and experts[i] != experts[i-1]+1:
                print ("problem")


        # Choose K experts at random, among the ones who report for all 267 matches
        np.random.seed()
        random.seed(sample_id)
        elf_experts = random.sample([i for i in range(14770, len(diff_forecasters))], num_experts)
        print ("Sampled Experts")
        print elf_experts


        num_matches = 0
        against     = []
        outcomes    = []

        for row in teams_sort:
            match = {'week' : row['week'], 'team1': row['team1'], 'team2' : row['team2']}
            if match not in against:
                against.append(match)
                num_matches += 1
                outcomes.append(float(row['game_outcome']))

        T = num_matches
        experts_reports = [[0 for _ in range(T)] for _ in range(num_experts)]
                
        for row in teams_sort:
            match = {'week' : row['week'], 'team1': row['team1'], 'team2' : row['team2']}
            game_index = against.index(match)
            if int(row['user_id']) in elf_experts:
                expert_index = elf_experts.index(int(row['user_id']))
                experts_reports[expert_index][game_index]  = float(row['user_prob']) 

    else:
        T = 2500
        num_experts     = 50
        outcomes        = [] 
        for t in range(T/2):
            outcomes.append(np.random.binomial(1, 0.4, 1))
        for t in range(T/2, T):
            outcomes.append(np.random.binomial(1, 0.6, 1))
    
        experts_reports = []
        
        for i in range(num_experts/3):
            temp = np.random.uniform(0, 0.7, T)
            experts_reports.append(temp)
        
        for i in range(num_experts/3, 2*num_experts/3):
            temp = np.random.uniform(0.3, 1.0, T)
            experts_reports.append(temp)
        
        for i in range(2*num_experts/3, num_experts):
            temp = np.random.uniform(0, 1.0, T)
            experts_reports.append(temp)
        

    return (num_experts, outcomes, experts_reports, T, num_repetitions)

