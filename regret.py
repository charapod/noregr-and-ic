# Module computing the regret for a sequence of actions.
import numpy as np

def regret(loss_lst, num_experts, algo_loss, T):
    loss_per_expert = []
    for i in range(num_experts):
        s = 0
        for t in range(T + 1):
            s += loss_lst[t][i]
        loss_per_expert.append(s)

    tot_algo_loss = sum(algo_loss)
    min_loss_hindsight = np.min(loss_per_expert)
    print ("Algorithm's Loss: %f"%(tot_algo_loss))
    print ("Best fixed: %f"%min_loss_hindsight)
    print ("Regret:%f"%(tot_algo_loss - min_loss_hindsight))
    
    # returns (regret to best, best_fixed loss)
    return (tot_algo_loss - min_loss_hindsight, min_loss_hindsight) 

