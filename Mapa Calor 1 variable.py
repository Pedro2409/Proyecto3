import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Lee la base de datos desde un archivo Excel
df = pd.read_csv('BD_madre.csv')

# Calcula la matriz de correlación con respecto a la columna 'punt_global'
correlation_matrix = df.corr()['punt_ingles']

# Crea un mapa de calor con Seaborn para la correlación con 'punt_global'
plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix.to_frame(), annot=True, cmap='coolwarm', linewidths=.5, cbar=True, vmin=-1, vmax=1)
plt.title('Correlación con respecto a punt_ingles')
plt.show()


# Lee la base de datos desde un archivo Excel
df = pd.read_csv('BD_madre.csv')

# Calcula la matriz de correlación con respecto a la columna 'punt_global'
correlation_matrix = df.corr()['punt_matematicas']

# Crea un mapa de calor con Seaborn para la correlación con 'punt_global'
plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix.to_frame(), annot=True, cmap='coolwarm', linewidths=.5, cbar=True, vmin=-1, vmax=1)
plt.title('Correlación con respecto a punt_matematicas')
plt.show()

# Lee la base de datos desde un archivo Excel
df = pd.read_csv('BD_madre.csv')

# Calcula la matriz de correlación con respecto a la columna 'punt_global'
correlation_matrix = df.corr()['punt_sociales_ciudadanas']

# Crea un mapa de calor con Seaborn para la correlación con 'punt_global'
plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix.to_frame(), annot=True, cmap='coolwarm', linewidths=.5, cbar=True, vmin=-1, vmax=1)
plt.title('Correlación con respecto a punt_sociales_ciudadanas')
plt.show()

# Lee la base de datos desde un archivo Excel
df = pd.read_csv('BD_madre.csv')

# Calcula la matriz de correlación con respecto a la columna 'punt_global'
correlation_matrix = df.corr()['punt_c_naturales']

# Crea un mapa de calor con Seaborn para la correlación con 'punt_global'
plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix.to_frame(), annot=True, cmap='coolwarm', linewidths=.5, cbar=True, vmin=-1, vmax=1)
plt.title('Correlación con respecto a punt_c_naturales')
plt.show()


# Calcula la matriz de correlación con respecto a la columna 'punt_global'
correlation_matrix = df.corr()['punt_lectura_critica']

# Crea un mapa de calor con Seaborn para la correlación con 'punt_global'
plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix.to_frame(), annot=True, cmap='coolwarm', linewidths=.5, cbar=True, vmin=-1, vmax=1)
plt.title('Correlación con respecto a punt_lectura_critica')
plt.show()
