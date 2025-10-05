from desertseeker.trial import Trial
from desertseeker.scso import SCSO
import random
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import warnings 
warnings.filterwarnings("ignore")

class Study:
    def __init__(self, 
                 pop_size= 10, 
                 iteration= 10,
                 early_stopping= False):
        self.trials= []
        self.pop_size= pop_size
        self.iteration= iteration
        self.early_stopping= early_stopping
        self.scso= SCSO(pop_size= self.pop_size, iteration= self.iteration)
    
    def compute_fitness(self, scores):
        # converting
        scores= np.array(scores)

        # delete inf and nan values
        scores = scores[~np.isnan(scores).any(axis=1)]
        scores = scores[~np.isinf(scores).any(axis=1)]

        # get minimum values
        min_index = np.argmin(scores[:, 0])

        return min_index
    
    def optimize(self, objective_function):
        # first population
        values, scores, parameters= self.scso.first_population(objective_function)
        
        # selecting best value
        max_index= self.compute_fitness(scores)
        max_value= values[max_index]
        max_score= scores[max_index]
        print(f"best value {max_value} with score {max_score}")
        
        # start trial
        counter= 0
        for i in range(self.iteration):
            # optimize SCSO
            result= self.scso.optimize(values= values,
                                       best_values= max_value,
                                       n_trial= self.iteration, 
                                       c_trial= i,
                                       parameters=parameters)
            
            # result value
            values= []
            scores= []
            for j in range(len(result)):
                trial= Trial(dynamic= False, static_values= result[j])
                scores.append(objective_function(trial))
                values.append(trial.values)
        
            # update best value
            current_best_index = self.compute_fitness(scores)
            current_best_value = values[current_best_index]
            current_best_score = scores[current_best_index]
            if (current_best_score < np.array(max_score)).all():
                print("update best value")
                max_value = current_best_value
                max_score = current_best_score
                
            print(f"Trial {i + 1} : best value {max_value} with score {max_score}")
            #print(values)
                
            # early stopping
            if self.early_stopping:   
                if max_value == values[max_index]:
                    counter+= 1
                    if counter == 5:
                        print("early stopped !")
                        return values[max_index], scores[max_index]
                
        
        return values[max_index], scores[max_index]
        
