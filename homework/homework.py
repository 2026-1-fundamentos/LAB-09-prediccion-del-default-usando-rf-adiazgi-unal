# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
# flake8: noqa: E501
import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# Directorio raíz del repositorio (un nivel arriba de homework/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Paso 1. Limpieza de datos
# ---------------------------------------------------------------------------

def load_and_clean(path):
    df = pd.read_csv(path)

    # Renombrar columna objetivo
    df = df.rename(columns={"default payment next month": "default"})

    # Remover columna ID
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Eliminar registros con información no disponible (SEX=0, MARRIAGE=0, EDUCATION=0)
    df = df[df["SEX"] != 0]
    df = df[df["MARRIAGE"] != 0]
    df = df[df["EDUCATION"] != 0]

    # Agrupar EDUCATION > 4 en categoría "others" (4)
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

    return df


train_df = load_and_clean(os.path.join(BASE_DIR, "files", "input", "train_data.csv.zip"))
test_df  = load_and_clean(os.path.join(BASE_DIR, "files", "input", "test_data.csv.zip"))


# ---------------------------------------------------------------------------
# Paso 2. División en X / y
# ---------------------------------------------------------------------------

TARGET = "default"

x_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

x_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]


# ---------------------------------------------------------------------------
# Paso 3. Pipeline
# ---------------------------------------------------------------------------

categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
numerical_features   = [c for c in x_train.columns if c not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="passthrough",
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier",   RandomForestClassifier(random_state=42)),
    ]
)


# ---------------------------------------------------------------------------
# Paso 4. Optimización de hiperparámetros con validación cruzada
# ---------------------------------------------------------------------------

param_grid = {
    "classifier__n_estimators":      [100, 200],
    "classifier__max_depth":         [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
}

cv = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring="balanced_accuracy",
    n_jobs=-1,
    refit=True,
)

cv.fit(x_train, y_train)
best_model = cv.best_estimator_

print("Mejores hiperparámetros:", cv.best_params_)


# ---------------------------------------------------------------------------
# Paso 5. Guardar el modelo comprimido
# ---------------------------------------------------------------------------

os.makedirs(os.path.join(BASE_DIR, "files", "models"), exist_ok=True)

with gzip.open(os.path.join(BASE_DIR, "files", "models", "model.pkl.gz"), "wb") as f:
    pickle.dump(best_model, f)


# ---------------------------------------------------------------------------
# Paso 6 & 7. Métricas y matrices de confusión
# ---------------------------------------------------------------------------

os.makedirs(os.path.join(BASE_DIR, "files", "output"), exist_ok=True)


def compute_metrics(model, x, y, dataset_name):
    y_pred = model.predict(x)

    metrics = {
        "dataset":           dataset_name,
        "precision":         precision_score(y, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "recall":            recall_score(y, y_pred, zero_division=0),
        "f1_score":          f1_score(y, y_pred, zero_division=0),
    }

    cm = confusion_matrix(y, y_pred)
    cm_entry = {
        "type":    "cm_matrix",
        "dataset": dataset_name,
        "true_0":  {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1":  {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
    }

    return metrics, cm_entry


train_metrics, train_cm = compute_metrics(best_model, x_train, y_train, "train")
test_metrics,  test_cm  = compute_metrics(best_model, x_test,  y_test,  "test")

with open(os.path.join(BASE_DIR, "files", "output", "metrics.json"), "w") as f:
    f.write(json.dumps(train_metrics) + "\n")
    f.write(json.dumps(test_metrics)  + "\n")
    f.write(json.dumps(train_cm)      + "\n")
    f.write(json.dumps(test_cm)       + "\n")

print("✓ Modelo guardado en files/models/model.pkl.gz")
print("✓ Métricas guardadas en files/output/metrics.json")
print("Train metrics:", train_metrics)
print("Test  metrics:", test_metrics)
