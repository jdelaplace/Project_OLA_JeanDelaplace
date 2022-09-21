from copy import copy,deepcopy
import numpy.random as npr
import numpy as np


class Simulator():
    def __init__(self,prices,margins,lamb) -> None:
        # self.prices are the price levels set for each product
        self.prices=prices
        # margins matrix associated for each product and each price point
        self.margins=margins
        self.lamb=lamb
        self.visited_primaries = []

    def simulation(self,j,user_class):
        # This recursive method simulates one user landing on a webpage of one product. 
        # Rewards depends on conversion rates, price point, number of items bought, and margins.
        # After adding to the rewards the money for the primary product, 
        # this method looks into the graph weights to give the most probable
        # secondary products and calls itself recursively to add the rewards of the next primary.

        
        # Compute reward for buying the primary
        history=np.zeros(5)
        history[j]=1
        history=np.array([history])
        rewards = np.zeros(5)
        rewards[j] = self.margins[j]*user_class.n_items_bought[j]*user_class.conv_rates[j]
      
        self.visited_primaries.append(j)

        arr = deepcopy(user_class.graph_weights)[j]
        arr[self.visited_primaries]=0.0
        # Select 2 secondaries with highest observation rates and visit them if weights are positive.
        # If they are null, means they have already been visited.
        first_secondary = np.argmax(arr)
        if arr[first_secondary]>npr.random():
            simu=self.simulation(first_secondary,user_class)
            history=np.concatenate((history,simu[1]),axis=0)
            rewards+=simu[0]
        arr[self.visited_primaries]=0.0
        arr1=deepcopy(arr)
        arr1[first_secondary]=0.0
        second_secondary = np.argmax(arr1)
        if arr[second_secondary]*self.lamb>npr.random():
            simu2=self.simulation(second_secondary,user_class)
            history=np.concatenate((history,simu2[1]),axis=0)
            rewards+=simu2[0]
        
        # Returns the rewards of that user associated to the products bought
        return rewards,history






        

