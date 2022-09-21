import numpy.random as npr
from simulator import *
from users import *
from website_simulation import website_simulation
from UCB import *
import matplotlib.pyplot as plt
from copy import deepcopy

def sample_alp(L,alp):
    L.append(alp)
    return L



if __name__=="__main__":
    # PARAMETERS TO CHOOSE ARBITRARILY
    graph_weights=npr.random((5,5))

    n_items_bought=npr.randint(1,5,size=5)
    v=np.array([0.8,0.6,0.4,0.2]).reshape((4,1))
    conv_rates=npr.normal(1,0.2,(4,5))*np.hstack((v,v,v,v,v))
    prices=np.arange(1.0,21.0).reshape((4,5))
    margins=np.arange(30.0,70.0,2.0).reshape((4,5))
    np.sort(prices,axis=0)
    lamb=0.2
    agl=ucb(5,4)

    #computation of the regret 
    T=300
    prices_init=[1.,1.,1.,1.,1.]
    sim = Simulator(prices_init,margins,lamb)
    total_reward=[]
    #clair_rewards=np.multiply(np.multiply(margins,n_items_bought),conv_rates)
    #opt=np.max(clair_rewards,axis=0)
    cumulative_regret=[]
    R=[]
    sample_alphas=[]
    sample_nb_items=[]
    #for n in range (1,N_exp):
    instant_regret=[]
    #initialization of the ucb by playing twice each price
    for j in range(0,8):
        k=j%2
        alpha_ratios=npr.dirichlet([20,20,20,20,20,20])
        sample_alp(sample_alphas,alpha_ratios)
        total_users=npr.normal(500,10)
        users= Users_group(total_users,alpha_ratios,graph_weights,n_items_bought,conv_rates)
        prices_pulled=[k,k,k,k,k]
        sim.prices=prices_pulled
        [rew,opt]=website_simulation(sim,users)
        #instant_regret.append(np.sum(opt-rew))
        instant_regret.append(np.sum(opt-rew))
        #total_reward.append(np.cumsum(rew))
        agl.update(prices_pulled,rew,j+1)
    for i in range (8,T):
        alpha_ratios=npr.dirichlet([20,20,20,20,20,20])
        sample_alp(sample_alphas,alpha_ratios)  #We store the observed alphas ratios each day
        total_users=npr.normal(500,10)
        users= Users_group(total_users,alpha_ratios,graph_weights,n_items_bought,conv_rates)
        prices_pulled=agl.act()
        sim.prices=prices_pulled
        [rew,opt]=website_simulation(sim,users)
        instant_regret.append(np.sum(opt-rew))
        agl.update(prices_pulled,rew,i)
    cumulative_regret=np.cumsum(instant_regret)
    est_alphas=np.mean(sample_alphas,axis=0) #We compute the mean of the sampled values of the alphas ratios
    #plt.plot(instant_regret)
    #plt.show()
    #best_indices=agl.act()
    #print('The best prices for each product are:',prices[best_indices,[0,1,2,3,4]])
    #print('The cumulative regret is',instant_regret)
    print('The estimation of the expected values of the alphas parameters is',est_alphas)
    print('The true expected value were',[0.16666667,0.16666667,0.16666667,0.16666667,0.16666667])
