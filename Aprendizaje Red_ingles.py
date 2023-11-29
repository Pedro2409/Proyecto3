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


datos = 'BD_ingles.csv'
df = pd.read_csv(datos)

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


#Se escoje el mejor modelo
modelo = BayesianNetwork(('periodo', 'fami_tieneinternet'), ('periodo', 'cole_area_ubicacion'), ('cole_area_ubicacion', 'fami_tieneinternet'), ('cole_area_ubicacion', 'punt_ingles'), ('cole_area_ubicacion', 'fami_tienecomputador'), ('cole_calendario', 'periodo'), ('cole_caracter', 'cole_mcpio_ubicacion'), ('cole_caracter', 'cole_naturaleza'), ('cole_caracter', 'cole_area_ubicacion'), ('cole_caracter', 'punt_ingles'), ('cole_mcpio_ubicacion', 'punt_ingles'), ('cole_mcpio_ubicacion', 'fami_tienecomputador'), ('cole_mcpio_ubicacion', 
'fami_tieneinternet'), ('cole_mcpio_ubicacion', 'cole_naturaleza'), ('cole_mcpio_ubicacion', 'cole_area_ubicacion'), ('cole_naturaleza', 'fami_tieneautomovil'), ('cole_naturaleza', 'fami_tienecomputador'), ('cole_naturaleza', 'fami_personashogar'), ('cole_naturaleza', 'cole_calendario'), ('cole_naturaleza', 'punt_ingles'), ('cole_naturaleza', 'cole_area_ubicacion'), ('cole_naturaleza', 'periodo'), ('fami_tienecomputador', 'fami_tieneinternet'), ('fami_tienecomputador', 'fami_tieneautomovil'), ('fami_tienecomputador', 'estu_tipodocumento'), ('fami_tieneinternet', 'fami_tieneautomovil'), ('fami_tieneinternet', 'fami_personashogar'), ('punt_ingles', 'estu_tipodocumento'), ('punt_ingles', 'fami_tienecomputador'), ('punt_ingles', 'fami_tieneautomovil'), ('punt_ingles', 'fami_personashogar'))
print(modelo)

# Obtén las variables sin predecesores
variables_sin_predecesores = [node for node in modelo.nodes() if not modelo.get_parents(node)]

# Imprime las variables sin predecesores
print("Variables sin predecesores:", variables_sin_predecesores)

sample_train, sample_test = train_test_split(df, test_size=0.2, random_state=77)

emv = MaximumLikelihoodEstimator(model = modelo, data = sample_train)

cpdem_m = emv.estimate_cpd(node='cole_caracter')
print(cpdem_m)

cpdem_ing = emv.estimate_cpd(node="punt_ingles")
print(cpdem_ing)


#METRICAS

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

target_columns = ["punt_ingles"]
# Create an empty dictionary to store predictions for each column
predictions = {}

# Iterate over each target column
for column in target_columns:
    # Separate features (X) and target variable (y) for train and test sets
    X_train = sample_train.drop(target_columns, axis=1)
    y_train = sample_train[column]
    X_test = sample_test.drop(target_columns, axis=1)
    y_test = sample_test[column]
    
    # Make predictions for the current target column
    y_pred = modelo.predict(X_test)
    
    # Store the predictions in the dictionary
    predictions[column] = y_pred

# Evaluate the overall performance
# Assuming you have the true test values in 'true_values'
overall_predictions = np.column_stack(list(predictions.values()))
overall_true_values = sample_test[target_columns].values

# Calculate classification report for each target column using overall_predictions
for i, column in enumerate(target_columns):
    y_true = sample_test[column]  # True labels for the current column
    y_pred = overall_predictions[:, i]  # Predictions for the current column
    
    # Calculate confusion matrix
    matriz = confusion_matrix(y_true, y_pred)
    
    # Print confusion matrix and classification report
    print(f"Confusion matrix for '{column}':")
    print(matriz)
    
    print(f"\nClassification report for '{column}':")
    report = classification_report(y_true, y_pred)
    print(report)
    print("-------------------------------")


modelo.fit(data=sample_train, estimator=MaximumLikelihoodEstimator)


infer = VariableElimination(modelo) 

# Se extraen los valores reales
y_real = sample_test["punt_ingles"].values

df2 = sample_test.drop(columns=['punt_ingles'])
y_p = modelo.predict(df2)

accuracy = accuracy_score(y_real, y_p)
print(accuracy)


conf = confusion_matrix(y_real, y_p)
print(conf)










