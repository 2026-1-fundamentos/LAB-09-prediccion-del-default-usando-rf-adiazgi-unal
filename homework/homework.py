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
import pickle
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
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

# -----------------------------------------------------------------------
# Rutas
# -----------------------------------------------------------------------
INPUT_DIR = Path("files/input")
MODELS_DIR = Path("files/models")
OUTPUT_DIR = Path("files/output")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------
# Paso 1: Limpieza de los datasets
# -----------------------------------------------------------------------
def load_zip_csv(zip_path: Path) -> pd.DataFrame:
    """Lee un csv que viene comprimido dentro de un .zip."""
    with zipfile.ZipFile(zip_path) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            return pd.read_csv(f)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Renombrar columna objetivo
    df = df.rename(columns={"default payment next month": "default"})

    # Remover columna ID
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Eliminar registros con informacion no disponible (codificada como 0
    # en EDUCATION y MARRIAGE en este dataset)
    df = df.loc[df["MARRIAGE"] != 0]
    df = df.loc[df["EDUCATION"] != 0]

    # Eliminar filas con NA explícitos, si existieran
    df = df.dropna()

    # Agrupar EDUCATION > 4 en la categoria "others" (4)
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

    return df


train_df = clean_dataset(load_zip_csv(INPUT_DIR / "train_data.csv.zip"))
test_df = clean_dataset(load_zip_csv(INPUT_DIR / "test_data.csv.zip"))


# -----------------------------------------------------------------------
# Paso 2: Dividir en x_train, y_train, x_test, y_test
# -----------------------------------------------------------------------
x_train = train_df.drop(columns=["default"])
y_train = train_df["default"]

x_test = test_df.drop(columns=["default"])
y_test = test_df["default"]


# -----------------------------------------------------------------------
# Paso 3: Pipeline (one-hot-encoding + random forest)
# -----------------------------------------------------------------------
categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ],
    remainder="passthrough",
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)


# -----------------------------------------------------------------------
# Paso 4: Optimizacion de hiperparametros con validacion cruzada
# -----------------------------------------------------------------------
param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [None, 5, 10, 20],
    "classifier__min_samples_split": [2, 5, 10],
}

model = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring="balanced_accuracy",
    n_jobs=-1,
    refit=True,
)

model.fit(x_train, y_train)


# -----------------------------------------------------------------------
# Paso 5: Guardar el modelo comprimido con gzip
# -----------------------------------------------------------------------
with gzip.open(MODELS_DIR / "model.pkl.gz", "wb") as f:
    pickle.dump(model, f)


# -----------------------------------------------------------------------
# Paso 6: Metricas de precision, precision balanceada, recall y f1-score
# -----------------------------------------------------------------------
def compute_metrics(dataset_name: str, y_true, y_pred) -> dict:
    return {
        "dataset": dataset_name,
        "precision": precision_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }


y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

train_metrics = compute_metrics("train", y_train, y_train_pred)
test_metrics = compute_metrics("test", y_test, y_test_pred)


# -----------------------------------------------------------------------
# Paso 7: Matrices de confusion
# -----------------------------------------------------------------------
def compute_cm_matrix(dataset_name: str, y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {
            "predicted_0": int(cm[0][0]),
            "predicted_1": int(cm[0][1]),
        },
        "true_1": {
            "predicted_0": int(cm[1][0]),
            "predicted_1": int(cm[1][1]),
        },
    }


train_cm = compute_cm_matrix("train", y_train, y_train_pred)
test_cm = compute_cm_matrix("test", y_test, y_test_pred)


# -----------------------------------------------------------------------
# Guardar metrics.json (una linea JSON por diccionario)
# -----------------------------------------------------------------------
with open(OUTPUT_DIR / "metrics.json", "w") as f:
    for record in [train_metrics, test_metrics, train_cm, test_cm]:
        f.write(json.dumps(record) + "\n")