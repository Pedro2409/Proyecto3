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


datos = 'BD_lectura.csv'
df = pd.read_csv(datos)

#modelo = BayesianNetwork([('estu_genero', 'punt_global'), ('fami_educacionmadre', 'punt_global'), ('fami_educacionpadre', 'punt_global'), ('fami_estratovivienda', 'punt_global'), ('fami_tienecomputador', 'punt_global'), ('fami_tieneinternet', 'punt_global')])
#modelo = BayesianNetwork([('periodo', 'fami_tieneinternet'), ('cole_area_ubicacion', 'fami_tieneinternet'), ('cole_area_ubicacion', 'punt_ingles'), ('cole_area_ubicacion', 'fami_tienecomputador'), ('cole_area_ubicacion', 'periodo'), ('cole_area_ubicacion', 'fami_personashogar'), ('cole_calendario', 'periodo'), ('cole_calendario', 'fami_tieneautomovil'), ('cole_caracter', 'cole_mcpio_ubicacion'), ('cole_caracter', 'cole_naturaleza'), ('cole_caracter', 'cole_area_ubicacion'), ('cole_caracter', 'punt_ingles'), ('cole_mcpio_ubicacion', 'fami_tienecomputador'), ('cole_mcpio_ubicacion', 'cole_naturaleza'), ('cole_mcpio_ubicacion', 'punt_ingles'), ('cole_mcpio_ubicacion', 'fami_tieneinternet'), ('cole_mcpio_ubicacion', 'fami_tieneautomovil'), ('cole_mcpio_ubicacion', 'cole_area_ubicacion'), ('cole_naturaleza', 'punt_ingles'), ('cole_naturaleza', 'fami_tieneautomovil'), ('cole_naturaleza', 'fami_tienecomputador'), ('cole_naturaleza', 'fami_personashogar'), ('cole_naturaleza', 'cole_area_ubicacion'), ('cole_naturaleza', 'cole_calendario'), ('fami_tienecomputador', 'fami_tieneinternet'), ('fami_tienecomputador', 'periodo'), ('fami_tienecomputador', 'estu_tipodocumento'), ('fami_tienecomputador', 'fami_tieneautomovil'), ('fami_tieneinternet', 'fami_personashogar'), ('punt_ingles', 'estu_tipodocumento'), ('punt_ingles', 'periodo'), ('punt_ingles', 'fami_tienecomputador'), ('punt_ingles', 'fami_personashogar'), ('punt_ingles', 'cole_calendario')])

modelo = BayesianNetwork([('periodo', 'fami_tieneinternet'), ('periodo', 'cole_area_ubicacion'), ('periodo', 'cole_caracter'), ('cole_area_ubicacion', 'fami_tieneinternet'), ('cole_area_ubicacion', 'fami_tienecomputador'), ('cole_area_ubicacion', 'punt_lectura_critica'), ('cole_area_ubicacion', 'fami_personashogar'), ('cole_caracter', 'cole_mcpio_ubicacion'), ('cole_caracter', 'cole_naturaleza'), ('cole_caracter', 'cole_area_ubicacion'), ('cole_caracter', 'punt_lectura_critica'), ('cole_caracter', 'fami_tieneautomovil'), ('cole_mcpio_ubicacion', 'fami_tienecomputador'), ('cole_mcpio_ubicacion', 'punt_lectura_critica'), ('cole_mcpio_ubicacion', 'cole_naturaleza'), ('cole_mcpio_ubicacion', 'fami_tieneinternet'), ('cole_mcpio_ubicacion', 'fami_tieneautomovil'), ('cole_mcpio_ubicacion', 'cole_area_ubicacion'), ('cole_naturaleza', 'fami_tieneautomovil'), ('cole_naturaleza', 'punt_lectura_critica'), ('cole_naturaleza', 'fami_tienecomputador'), ('cole_naturaleza', 'fami_personashogar'), ('cole_naturaleza', 'cole_area_ubicacion'), ('fami_tieneautomovil', 'fami_tienecomputador'), ('fami_tienecomputador', 'fami_tieneinternet'), ('fami_tienecomputador', 'estu_tipodocumento'), ('fami_tieneinternet', 'fami_personashogar'), ('punt_lectura_critica', 'estu_tipodocumento'), ('punt_lectura_critica', 'fami_personashogar')])
sample_train, sample_test = train_test_split(df, test_size=0.2, random_state=777)
emv = MaximumLikelihoodEstimator(model = modelo, data = sample_train)

modelo.fit(data=sample_train, estimator=MaximumLikelihoodEstimator)

infer = VariableElimination(modelo) 

# Se extraen los valores reales
y_real = sample_test["punt_lectura_critica"].values

df2 = sample_test.drop(columns=['punt_lectura_critica'])
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
    scoring_method=scoring_method, max_indegree=4, max_iter=int(1e4)
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

