from trial import Trial
from scso import SCSO
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
                 iteration= 10):
        self.trials= []
        self.pop_size= pop_size
        self.iteration= iteration
        self.scso= SCSO(pop_size= self.pop_size, iteration= self.iteration)
    
    def crowding_distance(self, objectives):
        n_points= objectives.shape[0]
        distances= np.zeros(n_points)
        
        for m in range(objectives.shape[1]):
            sorted_indices= np.argsort(objectives[:, m])
            distances[sorted_indices[0]]= np.inf
            distances[sorted_indices[-1]]= np.inf
            
            if n_points > 2:
                max_val= np.max(objectives[:, m])
                min_val= np.min(objectives[:, m])
                if max_val != min_val:
                    for i in range(1, n_points-1):
                        distances[sorted_indices[i]] += (
                            objectives[sorted_indices[i+1], m] - objectives[sorted_indices[i-1], m]
                            ) / (max_val - min_val)
        return distances
    
    def compute_fitness(self, scores):
        # Non-Dominated sorting
        scores= np.array(scores)
        nds= NonDominatedSorting()
        fronts= nds.do(scores)

        front_0 = fronts[0]
        front_0_objectives = scores[front_0]
        
        # crowding distance
        if front_0_objectives.shape[0] > 2:
            crowding_distance= self.crowding_distance(front_0_objectives)
            finite_values = crowding_distance[~np.isinf(crowding_distance)]
            #max_index= crowding_distance.index(max(finite_values))
            max_index= np.where(crowding_distance == max(finite_values))
            return np.where((scores == front_0_objectives[max_index]).all(axis= 1))[0][0]
        else:
            return np.where((scores == front_0_objectives[np.random.choice(front_0_objectives.shape[0])]).all(axis= 1))[0][0]
    
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
        
            # early stopping
            '''max_index= self.select_best_with_roulette_wheel(scores, max_score)
            if max_value == values[max_index]:
                counter+= 1
                if counter == 5:
                    print("early stopped !")
                    return values[max_index], scores[max_index]'''
        
            # update best value
            max_index= self.compute_fitness(scores)
            max_value= values[max_index]
            max_score= scores[max_index]
            print(f"Trial {i + 1} : best value {max_value} with score {max_score}")
        
        return values[max_index], scores[max_index]
        
# test Algorithm
def objective(trial):
    score= []
    df= pd.read_csv("Titanic-Dataset.csv")
    encoder= LabelEncoder()
    df= df.fillna(0)
    
    df['Sex']= encoder.fit_transform(df['Sex'])
    df['Embarked']= encoder.fit_transform(df['Embarked'])
    
    X= df.drop(columns= ['Survived'])
    y= df[['Survived']]
    
    x_train, x_test, y_train, y_test= train_test_split(X, y, test_size=0.2)

    model= RandomForestClassifier(
        n_estimators= trial.suggest_int('estimator', 100, 500),
        max_depth= trial.suggest_int('depth', 1, 50),
        )
    model.fit(x_train, y_train)
    
    pred= model.predict(x_test)
    
    acc= accuracy_score(y_test, pred)
    f1= f1_score(y_test, pred)
    
    score.append(acc)
    score.append(f1)
    
    return score
    
if __name__ == "__main__":
    study= Study(pop_size=50, iteration=15)
    score= study.optimize(objective)