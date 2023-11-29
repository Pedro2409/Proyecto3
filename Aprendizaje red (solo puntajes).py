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


datos = 'BD_Puntajes.xlsx'
df = pd.read_excel(datos)

modelo = BayesianNetwork([('punt_ingles', 'punt_global'), ('punt_matematicas', 'punt_global'), ('punt_sociales_ciudadanas', 'punt_global'), ('punt_c_naturales', 'punt_global'), ('punt_lectura_critica', 'punt_global')])

sample_train, sample_test = train_test_split(df, test_size=0.2, random_state=777)
emv = MaximumLikelihoodEstimator(model = modelo, data = sample_train)

modelo.fit(data=sample_train, estimator=MaximumLikelihoodEstimator)

infer = VariableElimination(modelo) 

# Se extraen los valores reales
y_real = sample_test["punt_global"].values

df2 = sample_test.drop(columns=['punt_global'])
y_p = modelo.predict(df2)

accuracy = accuracy_score(y_real, y_p)
print(accuracy)


conf = confusion_matrix(y_real, y_p)

# Extraer verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos