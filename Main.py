import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar los datos desde un archivo CSV
data = pd.read_csv('CreditoPersonal.csv')

# Suponiendo que la columna objetivo se llama 'target' y las características son todas las demás columnas
X = data.drop('target', axis=1)  # Características
y = data['target']  # Etiqueta