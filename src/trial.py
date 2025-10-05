import numpy as np
import random

class Trial:
    def __init__(self, dynamic= True, static_values= {}):
        self.parameters= {}                     # parameters
        self.values= {}                         # dynamic value
        self.dynamic= dynamic                   # create value or get static value
        self.static_values= static_values       # get static value
    
    # generate float number
    def suggest_float(self, name, low, high):
        # generate dynamic value
        if self.dynamic:
            if name not in self.parameters:
                self.parameters[name]= {'type' : 'float', 'low' : low, 'high' : high}
            else:
                raise Exception("the name of value is same with other value")
            # generate value
            value= np.random.uniform(low, high)
            self.values[name]= value
            return value
        # set static value
        else:
            value= self.static_values[name]
            self.values[name]= value
            return value
    
    # generate Int number
    def suggest_int(self, name, low, high):
        # generate dynamic value
        if self.dynamic:
            if name not in self.parameters:
                self.parameters[name]= {'type' : 'int', 'low' : low, 'high' : high}
            else:
                raise Exception("the name of value is same with other value")
            # generate value
            value= np.random.randint(low, high)
            self.values[name]= value
            return value
        # set static value
        else:
            value= self.static_values[name]
            self.values[name]= value
            return value
    
    def suggest_categorical(self, name, choices):
        # generate dynamic value
        if self.dynamic:
            if name not in self.parameters:
                self.parameters[name]= {'type' : 'categorical', 'choices' : choices}
            else:
                raise Exception("the name of value is same with other value")
            # generate value
            value= random.choice(choices)
            self.values[name]= value
            return value
        # set static value
        else:
            value= self.static_values[name]
            self.values[name]= value
            return value
        
        
if __name__ == '__main__':
    def objective(trial):
        x= trial.suggest_categorical('xy', [5.25, 10.75])
        return x + 1

    def first_population(pop_size, objective):
        values= []
        scores= []
        for _ in range(pop_size):
            # read values suggester
            trial= Trial()
            scores.append(objective(trial))
            values.append(trial.values)
            
        return values, scores, trial.parameters    

    values, scores, parameters= first_population(pop_size= 10, objective=objective)
