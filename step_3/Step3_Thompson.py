import numpy.random as npr
from users import *
import matplotlib.pyplot as plt
from copy import deepcopy

#Here is an adaptation of the Thompson Sampling code
          
class env_TS:
    def __init__(self,prices):
        self.prices=prices

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

class thompson(learner):
    def __init__(self, n_products,n_prices):
        super().__init__(n_products,n_prices)
        self.beta_parameters=np.ones((n_prices,n_products,2))
            
    def act(self):
        idx=np.argmax(np.random.beta(self.beta_parameters[:,:,0],self.beta_parameters[:,:,1]),axis=0)
        return idx

    def update(self,prices_pulled,reward):
            #price_pulled is a list of chosen prices for each product
            #as we have 5 products here it is a list of 5 prices
                super().update(prices_pulled,reward)
                for j in range (self.n_products):
                    self.beta_parameters[prices_pulled[j],j,0]=self.beta_parameters[prices_pulled[j],j,0]+reward[j]
                    self.beta_parameters[prices_pulled[j],j,1]=self.beta_parameters[prices_pulled[j],j,1]+1.0-reward[j]
                    
                


