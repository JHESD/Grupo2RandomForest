import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar los datos desde un archivo CSV
data = pd.read_csv('CreditoPersonal.csv')

# Suponiendo que la columna objetivo se llama 'target' y las características son todas las demás columnas
X = data.drop('target', axis=1)  # Características
y = data['target']  # Etiqueta

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
clf.fit(X_train, y_train)