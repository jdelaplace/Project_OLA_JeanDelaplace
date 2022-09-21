import numpy.random as npr
from simulator import *
from users import *
from website_simulation import website_simulation
from estim_weights import *
import matplotlib.pyplot as plt
from copy import deepcopy




if __name__=="__main__":
    # PARAMETERS TO CHOOSE ARBITRARILY
    graph_weights=npr.random((5,5))
    n_items_bought=npr.randint(1,5,size=5)
    conv_rates=npr.normal(0.5,0.2,size=5)
    prices=np.arange(1.0,6.0)
    margins=np.arange(10.0,60.0,step=10.0)
    lamb=0.2
    samples_weights=[]

    #We run T=10 simulations, then we give the mean of the values obtained
    T=30
    sim = Simulator(prices,margins,lamb)
    for i in range (1,T):
        #Each day, a new set of users visits the website
        alpha_ratios=npr.dirichlet([20,20,20,20,20,20])
        total_users=npr.normal(500,10)
        users= Users_group(total_users,alpha_ratios,graph_weights,n_items_bought,conv_rates)
        [rew,episodes]=website_simulation(sim,users)
        samples_episodes=[]
        for k in range (5): #We compute the estimation of the weights for each of the 5 products
            samples_episodes.append(estimate_weights(episodes,k,5))
        samples_weights.append(samples_episodes)
    estimated_weights=np.mean(samples_weights,axis=0)
    print('The estimated weights are',estimated_weights)
    print('The real weights are',graph_weights)