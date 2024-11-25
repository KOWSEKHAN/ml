import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import VariableElimination
model = BayesianNetwork([
    ('RestECG', 'HeartDisease'),
    ('ChestPain', 'HeartDisease')
])
rest_ecg_cpt = DiscreteFactor(variable=['RestECG'], 
                              outcomes=[0, 1, 2], 
                              probabilities=[0.4, 0.4, 0.2])
chest_pain_cpt = DiscreteFactor(variable=['ChestPain'], 
                                outcomes=['typical', 'asymptomatic', 'nonanginal'], 
                                probabilities=[0.5, 0.3, 0.2])
heart_disease_cpt = DiscreteFactor(variable=['HeartDisease'], 
                                   outcomes=[0, 1],  
                                   evidence=['RestECG', 'ChestPain'],
                                   probabilities=[[0.9, 0.1], 
                                                  [0.7, 0.3],  
                                                  [0.5, 0.5]])  
model.add_factors(rest_ecg_cpt, chest_pain_cpt, heart_disease_cpt)
inference = VariableElimination(model)
rest_ecg = int(input("Enter the Rest ECG value (0, 1, 2): "))
chest_pain = input("Enter the Chest Pain value (typical, asymptomatic, nonanginal): ")
result = inference.query(variables=['HeartDisease'], evidence={'RestECG': rest_ecg, 'ChestPain': chest_pain})
print(result)
