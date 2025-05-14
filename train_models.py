import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import mutual_info_regression
import os


# Asegurar directorios
os.makedirs("models", exist_ok=True)

# Cargar datos
df = pd.read_excel("dataset/FGR_dataset.xlsx")
df.dropna(inplace=True)

# Separar características y etiquetas
X = df.drop(columns=["C31"])
y = df["C31"]

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/scaler.pkl")

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

df_sample = df.sample(n=30, random_state=42)  # Puedes ajustar n
df_sample.to_csv("FGR_dataset_prueba.csv", index=False)

# 1. Regresión logística
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
joblib.dump(lr, "models/logistic_regression.pkl")

# 2. Red neuronal artificial
nn = MLPClassifier(hidden_layer_sizes=(30,), max_iter=1000)
nn.fit(X_train, y_train)
joblib.dump(nn, "models/neural_net.pkl")

# 3. SVM
svm = SVC(probability=True)
svm.fit(X_train, y_train)
joblib.dump(svm, "models/svm.pkl")

# 4. FCM

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

np.random.seed(42)
weights = np.random.uniform(-1, 1, size=(X_train.shape[1], X_train.shape[1])) 

joblib.dump(scaler, "models/escalador_fcm.pkl")
joblib.dump(weights, "models/pesos_fcm.pkl")

print("Todos los modelos entrenados y guardados con éxito.")
