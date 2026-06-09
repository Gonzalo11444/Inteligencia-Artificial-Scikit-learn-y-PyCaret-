import os
import time
import warnings
import joblib
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)


# ==========================
# CONFIGURACIÓN GENERAL
# ==========================

# Compatible con Google Colab (sin __file__) y ejecución local
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_DIR = os.getcwd()   # En Colab el directorio de trabajo es /content

DATA_PATH   = os.path.join(BASE_DIR, "data", "data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42


# ==========================
# 1. CARGA DEL DATASET
# ==========================

print("Cargando dataset...")

df = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8-sig")

# Limpiar nombres de columnas (espacios, tabuladores)
df.columns = df.columns.str.strip()

print(f"Dataset cargado correctamente: {df.shape[0]} filas y {df.shape[1]} columnas")


# ==========================
# 2. PREPARACIÓN DE LA VARIABLE OBJETIVO
# ==========================

# El dataset original tiene tres clases: Graduate, Dropout y Enrolled.
# Para este prototipo binario se eliminan los casos Enrolled.
# Graduate = Exito académico
# Dropout  = Riesgo académico

df = df[df["Target"].isin(["Graduate", "Dropout"])].copy()

df["Target_binary"] = df["Target"].map({
    "Graduate": "Exito",
    "Dropout":  "Riesgo"
})

df = df.drop(columns=["Target"])

print("Distribución de la variable objetivo:")
print(df["Target_binary"].value_counts())


# ==========================
# 3. REGISTRO DE INFORMACIÓN DEL DATASET
# ==========================

dataset_info_path = os.path.join(RESULTS_DIR, "dataset_info_sklearn.txt")

with open(dataset_info_path, "w", encoding="utf-8") as f:
    f.write("INFORMACIÓN DEL DATASET - PROTOTIPO A SCIKIT-LEARN\n")
    f.write("===================================================\n")
    f.write(f"Filas utilizadas: {df.shape[0]}\n")
    f.write(f"Columnas utilizadas: {df.shape[1]}\n")
    f.write("Variable objetivo: Target_binary\n")
    f.write("Clases utilizadas:\n")
    f.write(str(df["Target_binary"].value_counts()))
    f.write("\n\nNota: se han eliminado los registros con Target = Enrolled "
            "para convertir el problema en clasificación binaria.\n")


# ==========================
# 4. SEPARACIÓN DE VARIABLES Y SPLIT TRAIN/TEST
# ==========================

print("Dividiendo datos en entrenamiento y prueba...")

X = df.drop(columns=["Target_binary"])
y = df["Target_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Entrenamiento: {len(X_train)} muestras | Prueba: {len(X_test)} muestras")


# ==========================
# 5. COMPARACIÓN DE MODELOS
# ==========================

# Los mismos modelos que PyCaret evaluó automáticamente
MODELOS = {
    "lr":       ("Logistic Regression",              LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
    "gbc":      ("Gradient Boosting Classifier",     GradientBoostingClassifier(random_state=RANDOM_STATE)),
    "et":       ("Extra Trees Classifier",           ExtraTreesClassifier(random_state=RANDOM_STATE)),
    "rf":       ("Random Forest Classifier",         RandomForestClassifier(random_state=RANDOM_STATE)),
    "ada":      ("Ada Boost Classifier",             AdaBoostClassifier(random_state=RANDOM_STATE)),
    "ridge":    ("Ridge Classifier",                 RidgeClassifier()),
    "lda":      ("Linear Discriminant Analysis",     LinearDiscriminantAnalysis()),
    "svm":      ("SVM - Linear Kernel",              LinearSVC(random_state=RANDOM_STATE, max_iter=2000)),
    "qda":      ("Quadratic Discriminant Analysis",  QuadraticDiscriminantAnalysis()),
    "dt":       ("Decision Tree Classifier",         DecisionTreeClassifier(random_state=RANDOM_STATE)),
    "knn":      ("K Neighbors Classifier",           KNeighborsClassifier()),
    "nb":       ("Naive Bayes",                      GaussianNB()),
    "dummy":    ("Dummy Classifier",                 DummyClassifier()),
}

print(f"\nComparando {len(MODELOS)} modelos con validación cruzada (5 folds)...")
print("-" * 60)

resultados = []

for codigo, (nombre, clasificador) in MODELOS.items():
    pipeline = Pipeline(steps=[
        ("scaler",     StandardScaler()),
        ("classifier", clasificador)
    ])

    t0 = time.perf_counter()

    # Cross-validation con 5 folds (igual que PyCaret por defecto)
    acc_scores  = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    f1_scores   = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1_weighted")
    rec_scores  = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="recall_weighted")
    prec_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="precision_weighted")

    tt = round(time.perf_counter() - t0, 3)

    resultados.append({
        "Unnamed: 0": codigo,
        "Model":      nombre,
        "Accuracy":   round(acc_scores.mean(), 4),
        "Recall":     round(rec_scores.mean(), 4),
        "Prec.":      round(prec_scores.mean(), 4),
        "F1":         round(f1_scores.mean(), 4),
        "TT (Sec)":   tt,
    })

    print(f"  {nombre:<40} Acc={acc_scores.mean():.4f}  F1={f1_scores.mean():.4f}  ({tt}s)")

# Ordenar por F1 descendente (igual que PyCaret con sort="F1")
comparacion_df = pd.DataFrame(resultados).sort_values("F1", ascending=False).reset_index(drop=True)
comparacion_df.to_csv(os.path.join(RESULTS_DIR, "comparacion_modelos_sklearn.csv"), index=False)

# Seleccionar el mejor modelo según F1
mejor_codigo   = comparacion_df.iloc[0]["Unnamed: 0"]
mejor_nombre   = comparacion_df.iloc[0]["Model"]
mejor_clf      = MODELOS[mejor_codigo][1]

print(f"\n→ Mejor modelo seleccionado: {mejor_nombre}")


# ==========================
# 6. ENTRENAMIENTO DEL MEJOR MODELO
# ==========================

print(f"\nEntrenando '{mejor_nombre}' sobre todo el conjunto de entrenamiento...")

model = Pipeline(steps=[
    ("scaler",     StandardScaler()),
    ("classifier", mejor_clf)
])

train_start = time.perf_counter()
model.fit(X_train, y_train)
training_time = time.perf_counter() - train_start


# ==========================
# 7. PREDICCIÓN SOBRE CONJUNTO DE PRUEBA
# ==========================

print("Generando predicciones...")

prediction_start = time.perf_counter()
y_pred = model.predict(X_test)
prediction_time = time.perf_counter() - prediction_start


# ==========================
# 8. CÁLCULO DE MÉTRICAS
# ==========================

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="Riesgo")
recall    = recall_score(y_test, y_pred, pos_label="Riesgo")
f1        = f1_score(y_test, y_pred, pos_label="Riesgo")

metrics = {
    "accuracy":                accuracy,
    "precision_riesgo":        precision,
    "recall_riesgo":           recall,
    "f1_riesgo":               f1,
    "training_time_seconds":   training_time,
    "prediction_time_seconds": prediction_time,
    "total_time_seconds":      training_time + prediction_time,
    "random_state":            RANDOM_STATE
}

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(os.path.join(RESULTS_DIR, "metricas_sklearn.csv"), index=False)

print("\nMétricas del prototipo A:")
print(metrics_df.to_string(index=False))

# Classification report completo
report = classification_report(y_test, y_pred, target_names=["Exito", "Riesgo"])
with open(os.path.join(RESULTS_DIR, "classification_report_sklearn.txt"),
          "w", encoding="utf-8") as f:
    f.write("CLASSIFICATION REPORT - PROTOTIPO A SCIKIT-LEARN\n")
    f.write("=================================================\n")
    f.write(report)


# ==========================
# 9. MATRIZ DE CONFUSIÓN
# ==========================

cm = confusion_matrix(y_test, y_pred, labels=["Riesgo", "Exito"])

cm_df = pd.DataFrame(
    cm,
    index=["Real_Riesgo", "Real_Exito"],
    columns=["Pred_Riesgo", "Pred_Exito"]
)
cm_df.to_csv(os.path.join(RESULTS_DIR, "matriz_confusion_sklearn.csv"))

plt.figure()
plt.imshow(cm, cmap="Blues")
plt.title("Matriz de confusión - Scikit-learn")
plt.xticks([0, 1], ["Pred_Riesgo", "Pred_Exito"])
plt.yticks([0, 1], ["Real_Riesgo", "Real_Exito"])
plt.xlabel("Predicción")
plt.ylabel("Valor real")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "matriz_confusion_sklearn.png"))
plt.close()


# ==========================
# 10. GUARDAR MODELO
# ==========================

joblib.dump(model, os.path.join(RESULTS_DIR, "modelo_sklearn.pkl"))


# ==========================
# 11. GUARDAR RESUMEN DE TIEMPOS
# ==========================

with open(os.path.join(RESULTS_DIR, "tiempos_sklearn.txt"), "w", encoding="utf-8") as f:
    f.write("TIEMPOS DE EJECUCIÓN - PROTOTIPO A SCIKIT-LEARN\n")
    f.write("================================================\n")
    f.write(f"Tiempo de entrenamiento: {training_time:.4f} segundos\n")
    f.write(f"Tiempo de predicción:    {prediction_time:.4f} segundos\n")
    f.write(f"Tiempo total:            {training_time + prediction_time:.4f} segundos\n")


print("\nEjecución completada.")
print(f"Resultados guardados en: {RESULTS_DIR}")
