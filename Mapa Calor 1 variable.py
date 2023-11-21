import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Lee la base de datos desde un archivo Excel
df = pd.read_excel('BD_FINAL.xlsx')

# Calcula la matriz de correlación con respecto a la columna 'punt_global'
correlation_matrix = df.corr()['punt_global']

# Crea un mapa de calor con Seaborn para la correlación con 'punt_global'
plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix.to_frame(), annot=True, cmap='coolwarm', linewidths=.5, cbar=True, vmin=-1, vmax=1)
plt.title('Correlación con respecto a punt_global')
plt.show()
