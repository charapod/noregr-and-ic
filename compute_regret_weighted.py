'''
From the .txt's with the weighted losses and the best-fixed losses
produce files with the weighted regrets for each of the different expert samplings.
'''

def regret_weighted(T, num_repetitions, num_samples_K, num_experts):
    elf_weighted_regr = [[0.0]*T for _ in range(num_repetitions)]

    for i in range(num_samples_K):
        best_fixed = open("nfl18-19/%dexperts/best_fixed_losses_%dexpert-%d.txt"%(num_experts,num_experts, i), "r") 
        elf_org   = open("nfl18-19/%dexperts/elf_weighted_losses_%dexperts-%d.txt"%(num_experts,num_experts,i), "r")
        elf_fin   = open("nfl18-19/%dexperts/elf_weighted_regrets_%dexperts-%d.txt"%(num_experts,num_experts,i), "w")
        
        rep = 0
        for line in elf_org:
            elf_weighted_regr[rep] = [float(j) for j in line.split()]
            rep += 1
        
        rep = 0
        for line in best_fixed:
            bf_loss = [float(j) for j in line.split()]
            elf_weighted_regr[rep] = [elf_weighted_regr[rep][t] - bf_loss[t] for t in range(T)]

            s = ""
            for t in range(T):
                s += ("%.5f "%elf_weighted_regr[rep][t])
            s += "\n"
            elf_fin.write(s)
            rep += 1
                    
        elf_fin.close()
        elf_org.close()
    
    return


num_samples_K = 10
T = 267
num_repetitions = 50
for num_experts in range(5, 15, 5):
    regret_weighted(T, num_repetitions, num_samples_K, num_experts)

