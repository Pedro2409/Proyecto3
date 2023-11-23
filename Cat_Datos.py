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

datos = 'Datos_sin_vacios_excel.xlsx'
df = pd.read_excel(datos)

columnas=df.columns
print(columnas)


for i in columnas:
    result=df[i].unique()
    print(i,result)


# Definir el mapeo para estu_tipodocumento
tipo_documento_mapping = {'TI': 1, 'CC': 2, 'CE': 3, 'CR': 4, 'PEP': 5, 'NES': 6, 'PE': 7, 'CCB': 8, 'PPT': 9, 'PC': 10}
df['estu_tipodocumento'] = df['estu_tipodocumento'].map(tipo_documento_mapping)

# Definir el mapeo para cole_area_ubicacion
area_ubicacion_mapping = {'RURAL': 1, 'URBANO': 2}
df['cole_area_ubicacion'] = df['cole_area_ubicacion'].map(area_ubicacion_mapping)

bilingue_mapping = {'S': 1, 'N': 2}
calendario_mapping = {'A': 1, 'B': 2}
caracter_mapping = {'TÉCNICO/ACADÉMICO': 1, 'TÉCNICO': 2, 'ACADÉMICO': 3, 'NO APLICA':4}
jornada_mapping = {'TARDE': 1, 'MAÑANA': 2, 'NOCHE': 3, 'COMPLETA': 4, 'SABATINA': 5,'UNICA': 6}

df['cole_bilingue'] = df['cole_bilingue'].map(bilingue_mapping)
df['cole_calendario'] = df['cole_calendario'].map(calendario_mapping)
df['cole_caracter'] = df['cole_caracter'].map(caracter_mapping)
df['cole_jornada'] = df['cole_jornada'].map(jornada_mapping)

municipio_mapping = {
    'SABANAS DE SAN ÁNGEL': 1,
    'ZAPAYÁN': 2,
    'SANTA MARTA': 3,
    'PUEBLOVIEJO': 4,
    'PEDRAZA': 5,
    'SAN SEBASTIÁN DE BUENAVISTA': 6,
    'SANTA ANA': 7,
    'PIVIJAY': 8,
    'FUNDACIÓN': 9,
    'ZONA BANANERA': 10,
    'EL RETÉN': 11,
    'PLATO': 12,
    'EL BANCO': 13,
    'SAN ZENÓN': 14,
    'GUAMAL': 15,
    'CHIVOLO': 16,
    'ARIGUANÍ': 17,
    'SALAMINA': 18,
    'CIÉNAGA': 19,
    'CERRO DE SAN ANTONIO': 20,
    'SITIONUEVO': 21,
    'ALGARROBO': 22,
    'ARACATACA': 23,
    'SANTA BÁRBARA DE PINTO': 24,
    'REMOLINO': 25,
    'NUEVA GRANADA': 26,
    'PIJIÑO DEL CARMEN': 27,
    'CONCORDIA': 28,
    'TENERIFE': 29,
    'EL PIÑÓN': 30
}

df['cole_mcpio_ubicacion'] = df['cole_mcpio_ubicacion'].map(municipio_mapping)

naturaleza_mapping = {'OFICIAL': 1, 'NO OFICIAL': 2}
genero_mapping = {'M': 1, 'F': 2}
educacion_mapping = {
    'Educación profesional completa': 1,
    'Secundaria (Bachillerato) completa': 2,
    'Primaria incompleta': 3,
    'Primaria completa': 4,
    'Secundaria (Bachillerato) incompleta': 5,
    'Técnica o tecnológica incompleta': 6,
    'Técnica o tecnológica completa': 7,
    'No sabe': 8,
    'Postgrado': 9,
    'Ninguno': 10,
    'Educación profesional incompleta': 11,
    'No Aplica': 12
}

df['cole_naturaleza'] = df['cole_naturaleza'].map(naturaleza_mapping)
df['estu_genero'] = df['estu_genero'].map(genero_mapping)
df['fami_educacionmadre'] = df['fami_educacionmadre'].map(educacion_mapping)
df['fami_educacionpadre'] = df['fami_educacionpadre'].map(educacion_mapping)

estrato_mapping = {'Estrato 1': 1, 'Sin Estrato': 7, 'Estrato 3': 3, 'Estrato 2': 2, 'Estrato 5': 5, 'Estrato 6': 6, 'Estrato 4': 4}
personas_hogar_mapping = {'1 a 2': 1,'3 a 4': 2, '5 a 6': 3, '7 a 8': 4, '9 o más': 5}
tiene_automovil_mapping = {'No': 0, 'Si': 1}
tiene_computador_mapping = {'No': 0, 'Si': 1,}
tiene_internet_mapping = {'No': 0, 'Si': 1}
tiene_lavadora_mapping = {'No': 0, 'Si': 1}

df['fami_estratovivienda'] = df['fami_estratovivienda'].map(estrato_mapping)
df['fami_personashogar'] = df['fami_personashogar'].map(personas_hogar_mapping)
df['fami_tieneautomovil'] = df['fami_tieneautomovil'].map(tiene_automovil_mapping)
df['fami_tienecomputador'] = df['fami_tienecomputador'].map(tiene_computador_mapping)
df['fami_tieneinternet'] = df['fami_tieneinternet'].map(tiene_internet_mapping)
df['fami_tienelavadora'] = df['fami_tienelavadora'].map(tiene_lavadora_mapping)

lim_Notas = [-1, 30, 50, 70, 100]
labels = ['Malo', 'Regular', 'Buena', 'Excelente']

# Aplicar la categorización por rangos y reemplazar las etiquetas con valores numéricos
df['punt_ingles'] = pd.cut(df['punt_ingles'], bins=lim_Notas, labels=labels)
df['punt_ingles'] = df['punt_ingles'].replace({'Malo': 1, 'Regular': 2, 'Buena': 3, 'Excelente': 4})

df['punt_matematicas'] = pd.cut(df['punt_matematicas'], bins=lim_Notas, labels=labels)
df['punt_matematicas'] = df['punt_matematicas'].replace({'Malo': 1, 'Regular': 2, 'Buena': 3, 'Excelente': 4})

df['punt_sociales_ciudadanas'] = pd.cut(df['punt_sociales_ciudadanas'], bins=lim_Notas, labels=labels)
df['punt_sociales_ciudadanas'] = df['punt_sociales_ciudadanas'].replace({'Malo': 1, 'Regular': 2, 'Buena': 3, 'Excelente': 4})

df['punt_c_naturales'] = pd.cut(df['punt_c_naturales'], bins=lim_Notas, labels=labels)
df['punt_c_naturales'] = df['punt_c_naturales'].replace({'Malo': 1, 'Regular': 2, 'Buena': 3, 'Excelente': 4})

df['punt_lectura_critica'] = pd.cut(df['punt_lectura_critica'], bins=lim_Notas, labels=labels)
df['punt_lectura_critica'] = df['punt_lectura_critica'].replace({'Malo': 1, 'Regular': 2, 'Buena': 3, 'Excelente': 4})

lim_Punt = [0, 99, 199, 299, 399, 500]
labelsPunt = ['Insuficiente', 'Malo', 'Regular', 'Suficiente', 'Sobresaliente']

df['punt_global'] = pd.cut(df['punt_global'], bins=lim_Punt, labels=labelsPunt)
df['punt_global'] = df['punt_global'].replace({'Insuficiente': 1, 'Malo': 2, 'Regular': 3, 'Suficiente': 4, 'Sobresaliente':5})


columnas=df.columns
print(columnas)

for i in columnas:
    result=df[i].unique()
    print(i,result)

df.to_csv('BD_Mod_Fin.csv',index=False)

