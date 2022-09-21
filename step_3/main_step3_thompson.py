import numpy.random as npr
from simulator import *
from users import *
from website_simulation import website_simulation
from Step3_Thompson import *
import matplotlib.pyplot as plt
from copy import deepcopy



# PARAMETERS TO CHOOSE ARBITRARILY
if __name__=="__main__":
    
    graph_weights=npr.random((5,5))

    n_items_bought=npr.randint(1,5,size=5)
    v=np.array([0.8,0.6,0.4,0.2]).reshape((4,1))
    conv_rates=npr.normal(1,0.2,(4,5))*np.hstack((v,v,v,v,v))
    prices=np.arange(1.0,21.0).reshape((4,5))
    margins=np.arange(30.0,70.0,2.0).reshape((4,5))
    np.sort(prices,axis=0)
    lamb=0.2
    env=env_TS(prices)
    agl=thompson(5,4)

    #computation of the regret
    N_exp=10
    T=300
    prices_init=[1.,1.,1.,1.,1.]
    sim = Simulator(prices_init,margins,lamb)
    optimal_rewards=[]
    cumulative_regret=[]
    rewards=[]
    for k in range(N_exp):
        optimal=[]
        total_reward=[]
        instant_regret=[]
        for i in range (1,T):
            alpha_ratios=npr.dirichlet([20,20,20,20,20,20])
            total_users=npr.normal(500,10)
            users= Users_group(total_users,alpha_ratios,graph_weights,n_items_bought,conv_rates)
            prices_pulled=agl.act()
            sim.prices=prices_pulled
            [rew,opt]=website_simulation(sim,users)
            instant_regret.append(np.sum(opt-rew))
            optimal.append(np.sum(opt))
            norm=np.linalg.norm(rew)
            total_reward.append(np.sum(rew))
            #I normalize so that the rewards is between 0 and 1
            agl.update(prices_pulled,rew/norm)

        cumulative_regret.append(np.cumsum(instant_regret))
        rewards.append(np.cumsum(total_reward))
        optimal_rewards.append(np.cumsum(optimal))
    mean_reward=np.mean(rewards,axis=0)
    std_dev_reward=np.std(rewards,axis=0)/np.sqrt(N_exp)
    mean_cum_regret=np.mean(cumulative_regret,axis=0)
    std_dev_regret=np.std(cumulative_regret,axis=0)/np.sqrt(N_exp)

    best_indices=agl.act()
    print('The best prices for each product are:',prices[best_indices,[0,1,2,3,4]])
    plt.figure(0)
    plt.title("Mean rewards over time")
    plt.plot(mean_reward/np.arange(1,T),label="Total expected reward")
    plt.xlabel("Time")
    plt.ylabel("Rewards")

    plt.figure(1)
    plt.plot(mean_cum_regret)
    plt.title("Mean regret over time")
    plt.xlabel("Time")
    plt.ylabel("Cumulative regret")
    plt.legend()
    plt.show()