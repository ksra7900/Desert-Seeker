import numpy as np
from trial import Trial
import random
import math


class SCSO:
    def __init__(self, pop_size, iteration):
        self.pop_size= pop_size
        self.iteration= iteration
        self.SM= 2
        
    def first_population(self, objective_function):
        values= []
        scores= []
        parameters= []
        for _ in range(self.pop_size):
            # read values suggester
            trial= Trial()
            scores.append(objective_function(trial))
            values.append(trial.values)
            parameters.append(trial.parameters)
            
        return values, scores, parameters
    
    def roulette_wheel_cos(self):
        # select degree
        theta= random.uniform(0, 360)
        # compute cosine
        if theta >= 0 or theta <= 30:    
            return math.cos(math.radians(random.choice([0, 30])))
        elif theta > 30 or theta <= 45:    
            return math.cos(math.radians(random.choice([30, 45])))
        elif theta > 45 or theta <= 90:    
            return math.cos(math.radians(random.choice([45, 90])))
        elif theta > 90 or theta <= 90:    
            return math.cos(math.radians(random.choice([45, 90])))
        
    def optimize(self, values, best_values, n_trial, c_trial, parameters):
        # Global Sensitivity
        rg = self.SM - (2 * self.SM * c_trial / (n_trial + n_trial))
        
        # Private Sensitivity
        r= rg * random.random()
        
        # Checking Phase
        R= 2 * rg * random.random() - rg
        
        # explotation
        if R <= 1:
            print("exploitation")
            for pos in values:
                for key in pos:
                    # counter
                    i= 0
                    # compute distance
                    PosMd = np.abs(random.random() * best_values[key] - pos[key])
                    # update position
                    pos[key]= best_values[key] - PosMd * self.roulette_wheel_cos() * r
                    # check domain value
                    if parameters[i][key]['type'] in ['float', 'int']:
                        #pos[key]= np.clip(pos[key], parameters[i][key]['low'], parameters[i][key]['high']) 
                        if pos[key] > parameters[i][key]['high']:
                            pos[key]= parameters[i][key]['high']
                            
                        if pos[key] < parameters[i][key]['low']:
                            pos[key]= parameters[i][key]['low']
                    # check type
                    if parameters[i][key]['type'] == 'int':
                        pos[key]= int(pos[key])
                    i+= 1
                    
        # exploration
        else:
            for pos in values:
                for key in pos:
                    # counter
                    i= 0
                    # Update position
                    pos[key]= r * (best_values[key] - random.random() * pos[key])
                    # check domain value
                    if parameters[i][key]['type'] in ['float', 'int']:
                        #pos[key]= np.clip(pos[key], parameters[i][key]['low'], parameters[i][key]['high'])
                        if pos[key] > parameters[i][key]['high']:
                            pos[key]= parameters[i][key]['high']
                            
                        if pos[key] < parameters[i][key]['low']:
                            pos[key]= parameters[i][key]['low']
                    # check type
                    if parameters[i][key]['type'] == 'int':
                        pos[key]= int(pos[key])
                    i+= 1
        
        return values