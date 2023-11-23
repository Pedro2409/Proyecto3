import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Lee la base de datos desde un archivo Excel
df = pd.read_csv('Datos_sin_vacios.csv')

#Verificamos que el puntaje no tenga vacios

cont=0

for i in range (0, (len(df['punt_ingles'])-1)):
    if df["punt_ingles"] [i] == 'NaN':
        cont += 1

if cont == 0:
    print('No hay vacíos')
else: 
    print('Tiene vacíos')

# Calcula la matriz de correlación con respecto a la columna 'punt_global'
correlation_matrix = df.corr()['punt_ingles']

# Crea un mapa de calor con Seaborn para la correlación con 'punt_global'
plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix.to_frame(), annot=True, cmap='coolwarm', linewidths=.5, cbar=True, vmin=-1, vmax=1)
plt.title('Correlación con respecto a punt_ingles')
plt.show()

