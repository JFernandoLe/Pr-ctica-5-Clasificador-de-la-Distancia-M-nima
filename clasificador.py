import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ====== 1. Cargar dataset ======
df = pd.read_csv("vocal_classification_dataset.csv")

# ====== 2. Preprocesamiento ======

# Eliminar columnas no numéricas irrelevantes
df = df.drop(columns=["Participant_ID", "Task_Type"])

# Función para convertir valores con rangos a promedio
def rango_promedio(valor):
    if pd.isna(valor):
        return np.nan
    if "-" in str(valor):
        try:
            a, b = valor.split("-")
            return (float(a) + float(b)) / 2
        except:
            return np.nan
    else:
        try:
            return float(valor)
        except:
            return np.nan

# Aplicar la función a las columnas con rangos
df["Formant_Frequency_Range (Hz)"] = df["Formant_Frequency_Range (Hz)"].apply(rango_promedio)
df["Pitch_Range (Hz)"] = df["Pitch_Range (Hz)"].apply(rango_promedio)

# Llenar valores faltantes con la media
df = df.fillna(df.mean(numeric_only=True))

# ====== 3. Separar características y etiquetas ======
X = df.drop(columns=["Target"]).values
y = df["Target"].values

# ====== 4. Normalización ======
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ====== 5. Definición del Clasificador de Distancia Mínima ======
class MinimumDistanceClassifier:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centroids_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])

    def predict(self, X):
        distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids_])
        return self.classes_[np.argmin(distances, axis=0)]

# ====== 6. Validación Hold-Out (70/30) ======
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = MinimumDistanceClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("=== Hold-Out 70/30 ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

# ====== 7. Validación 10-Fold Cross Validation ======
kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

print("\n=== 10-Fold Cross Validation ===")
print("Accuracy promedio:", round(np.mean(accuracies), 3))
