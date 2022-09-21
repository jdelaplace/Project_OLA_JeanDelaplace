from re import S
import numpy.random as npr
from users import *
import matplotlib.pyplot as plt
from copy import deepcopy

#Here is an adaptation of the UCB code
# We have an optimization for 5 different items instead of 1
# So all the matrices have 2 dimensions (product and price) instead of one.
# The arms of the UCB algorithm are the prices of each item, that can take 4 different values.



class learner:
    def __init__(self,n_products,n_prices):
        self.n_products=n_products
        self.n_prices=n_prices
        self.collected_rewards=[]
        list=[[]for _ in range (n_products)]
        list2=[]
        for i in range (n_prices):
            list2.append(list)
        self.rewards=list2

    def act(self):
        pass

    def update(self,prices_pulled, reward): #prices_pulled is a list of the indices of the prices pulled for each product
        for k in range(self.n_products):
            self.rewards[prices_pulled[k]][k].append(reward[k])

class ucb(learner):
    def __init__(self, n_products,n_prices):
        super().__init__(n_products,n_prices)
        self.means=np.zeros((n_prices,n_products))
        self.widths=np.zeros((n_prices,n_products))
        for k in self.widths:
            k=np.inf
            
    def act(self):
        idx=np.argmax(self.means+self.widths,axis=0)
        return idx

    def update(self,prices_pulled,reward,t):
            #price_pulled is a list of chosen prices for each product
            #as we have 5 products here it is a list of 5 prices
            #reward is the list of the rewards for each product obtained for an iteration
                super().update(prices_pulled,reward)
                for j in range (self.n_products):
                    self.means[prices_pulled[j]][j]=np.mean(self.rewards[prices_pulled[j]][j])
                    for idx in range (self.n_prices):
                        n=len(self.rewards[idx][j])
                        if n>0:
                            self.widths[idx][j]=np.sqrt(2000000*np.log(t)/n)
                            #There a 2000000 in the sqrt so that the bound is at the same scale as the rewards
                        else: 
                            self.widths[idx][j]=np.inf
                


