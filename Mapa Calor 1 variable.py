import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Lee la base de datos desde un archivo Excel
df = pd.read_csv('BD_Mod_Fin.csv')

#Verificamos que los puntajes no tenga vacios

cont=0

for i in range (0, (len(df['punt_ingles'])-1)):
    if df["punt_ingles"] [i] == 'NaN':
        cont += 1

if cont == 0:
    print('No hay vacíos')
else: 
    print('Tiene vacíos')

cont=0

for i in range (0, (len(df['punt_matematicas'])-1)):
    if df["punt_matematicas"] [i] == 'NaN':
        cont += 1

if cont == 0:
    print('No hay vacíos')
else: 
    print('Tiene vacíos')

cont=0

for i in range (0, (len(df['punt_sociales_ciudadanas'])-1)):
    if df["punt_sociales_ciudadanas"] [i] == 'NaN':
        cont += 1

if cont == 0:
    print('No hay vacíos')
else: 
    print('Tiene vacíos')

cont=0

for i in range (0, (len(df['punt_c_naturales'])-1)):
    if df["punt_c_naturales"] [i] == 'NaN':
        cont += 1

if cont == 0:
    print('No hay vacíos')
else: 
    print('Tiene vacíos')

cont=0

for i in range (0, (len(df['punt_lectura_critica'])-1)):
    if df["punt_lectura_critica"] [i] == 'NaN':
        cont += 1

if cont == 0:
    print('No hay vacíos')
else: 
    print('Tiene vacíos')

cont=0

for i in range (0, (len(df['punt_global'])-1)):
    if df["punt_global"] [i] == 'NaN':
        cont += 1

if cont == 0:
    print('No hay vacíos')
else: 
    print('Tiene vacíos')

# Calcula la matriz de correlación con respecto a la columna 'punt_global'
correlation_matrix = df.corr()['punt_global']

# Crea un mapa de calor con Seaborn para la correlación con 'punt_global'
plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix.to_frame(), annot=True, cmap='coolwarm', linewidths=.5, cbar=True, vmin=-1, vmax=1)
plt.title('Correlación con respecto a punt_global')
plt.show()

