from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import BayesianEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import plotly.express as px
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np


datos = 'BD_sociales.csv'
df = pd.read_csv(datos)

#modelo = BayesianNetwork([('estu_genero', 'punt_global'), ('fami_educacionmadre', 'punt_global'), ('fami_educacionpadre', 'punt_global'), ('fami_estratovivienda', 'punt_global'), ('fami_tienecomputador', 'punt_global'), ('fami_tieneinternet', 'punt_global')])
modelo = BayesianNetwork([('cole_area_ubicacion', 'cole_naturaleza'), ('cole_area_ubicacion', 'periodo'), ('cole_area_ubicacion', 'punt_sociales_ciudadanas'), ('cole_mcpio_ubicacion', 'cole_area_ubicacion'), ('cole_mcpio_ubicacion', 'cole_naturaleza'), ('cole_mcpio_ubicacion', 'fami_tienecomputador'), ('cole_naturaleza', 'fami_tieneautomovil'), ('cole_naturaleza', 
'punt_sociales_ciudadanas'), ('cole_naturaleza', 'periodo'), ('fami_tienecomputador', 'fami_tieneautomovil'), ('fami_tienecomputador', 'cole_naturaleza'), ('fami_tienecomputador', 'periodo'), ('fami_tienecomputador', 'punt_sociales_ciudadanas'), ('fami_tienecomputador', 'estu_tipodocumento'), ('fami_tieneinternet', 'fami_tienecomputador'), 
('fami_tieneinternet', 'cole_mcpio_ubicacion'), ('fami_tieneinternet', 'cole_area_ubicacion'), ('fami_tieneinternet', 'punt_sociales_ciudadanas'), ('fami_tieneinternet', 'periodo'), ('fami_tieneinternet', 'fami_tieneautomovil'), ('punt_sociales_ciudadanas', 'estu_tipodocumento')])
sample_train, sample_test = train_test_split(df, test_size=0.2, random_state=777)
emv = MaximumLikelihoodEstimator(model = modelo, data = sample_train)

modelo.fit(data=sample_train, estimator=MaximumLikelihoodEstimator)

infer = VariableElimination(modelo) 

# Se extraen los valores reales
y_real = sample_test["punt_sociales_ciudadanas"].values

df2 = sample_test.drop(columns=['punt_sociales_ciudadanas'])
y_p = modelo.predict(df2)

accuracy = accuracy_score(y_real, y_p)
print(accuracy)


conf = confusion_matrix(y_real, y_p)




#Puntaje

from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score

scoring_method = K2Score(data=df)
esth = HillClimbSearch(data=df)
estimated_modelh = esth.estimate(
    scoring_method=scoring_method, max_indegree=4, max_iter=int(1e1)
)
print(estimated_modelh)
print(estimated_modelh.nodes())
print(estimated_modelh.edges())

#Mismo procedimiento con BicScore
from pgmpy.estimators import BicScore

scoring_method = BicScore(data=df)
esth = HillClimbSearch(data=df)
estimated_modelBicScore = esth.estimate(
    scoring_method=scoring_method, max_indegree=4, max_iter=int(1e4)
)
print(estimated_modelBicScore)
print(estimated_modelBicScore.nodes())
print(estimated_modelBicScore.edges())

print(scoring_method.score(estimated_modelBicScore))    
print(scoring_method.score(estimated_modelh))
