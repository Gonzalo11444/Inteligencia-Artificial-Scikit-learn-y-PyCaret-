import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pycaret.classification import setup, compare_models, predict_model, save_model, pull


# ==========================
# CONFIGURACIÓN GENERAL
# ==========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42


# ==========================
# 1. CARGA DEL DATASET
# ==========================

print("Cargando dataset...")

df = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8-sig")

# Limpiar nombres de columnas para evitar problemas con tabuladores o espacios raros
df.columns = df.columns.str.strip()

print(f"Dataset cargado correctamente: {df.shape[0]} filas y {df.shape[1]} columnas")


# ==========================
# 2. PREPARACIÓN DE LA VARIABLE OBJETIVO
# ==========================

# El dataset original tiene tres clases:
# Graduate, Dropout y Enrolled.
# Para este prototipo binario se eliminan los casos Enrolled.
# Graduate = Exito académico
# Dropout = Riesgo académico

df = df[df["Target"].isin(["Graduate", "Dropout"])].copy()

df["Target_binary"] = df["Target"].map({
    "Graduate": "Exito",
    "Dropout": "Riesgo"
})

df = df.drop(columns=["Target"])

print("Distribución de la variable objetivo:")
print(df["Target_binary"].value_counts())


# ==========================
# 3. REGISTRO DE INFORMACIÓN DEL DATASET
# ==========================

dataset_info_path = os.path.join(RESULTS_DIR, "dataset_info_pycaret.txt")

with open(dataset_info_path, "w", encoding="utf-8") as f:
    f.write("INFORMACIÓN DEL DATASET - PROTOTIPO B PYCARET\n")
    f.write("================================================\n")
    f.write(f"Filas utilizadas: {df.shape[0]}\n")
    f.write(f"Columnas utilizadas: {df.shape[1]}\n")
    f.write("Variable objetivo: Target_binary\n")
    f.write("Clases utilizadas:\n")
    f.write(str(df["Target_binary"].value_counts()))
    f.write("\n\nNota: se han eliminado los registros con Target = Enrolled para convertir el problema en clasificación binaria.\n")


# ==========================
# 4. CONFIGURACIÓN DE PYCARET
# ==========================

print("Configurando experimento de PyCaret...")

setup_start = time.perf_counter()

clf = setup(
    data=df,
    target="Target_binary",
    train_size=0.8,
    session_id=RANDOM_STATE,
    normalize=True,
    html=False,
    verbose=False
)

setup_end = time.perf_counter()
setup_time = setup_end - setup_start


# ==========================
# 5. ENTRENAMIENTO Y SELECCIÓN DEL MODELO
# ==========================

print("Entrenando y comparando modelos con PyCaret...")

train_start = time.perf_counter()

best_model = compare_models(sort="F1", n_select=1)

train_end = time.perf_counter()
training_time = train_end - train_start

# Guardar tabla de comparación de modelos generada por PyCaret
comparison_table = pull()
comparison_table.to_csv(os.path.join(RESULTS_DIR, "comparacion_modelos_pycaret.csv"), index=True)


# ==========================
# 6. PREDICCIÓN SOBRE CONJUNTO DE PRUEBA
# ==========================

print("Generando predicciones...")

prediction_start = time.perf_counter()

predictions = predict_model(best_model)

prediction_end = time.perf_counter()
prediction_time = prediction_end - prediction_start


# ==========================
# 7. CÁLCULO DE MÉTRICAS
# ==========================

y_true = predictions["Target_binary"]
y_pred = predictions["prediction_label"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label="Riesgo")
recall = recall_score(y_true, y_pred, pos_label="Riesgo")
f1 = f1_score(y_true, y_pred, pos_label="Riesgo")

metrics = {
    "accuracy": accuracy,
    "precision_riesgo": precision,
    "recall_riesgo": recall,
    "f1_riesgo": f1,
    "setup_time_seconds": setup_time,
    "training_time_seconds": training_time,
    "prediction_time_seconds": prediction_time,
    "total_time_seconds": setup_time + training_time + prediction_time,
    "random_state": RANDOM_STATE
}

metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(os.path.join(RESULTS_DIR, "metricas_pycaret.csv"), index=False)

print("Métricas del prototipo B:")
print(metrics_df)


# ==========================
# 8. MATRIZ DE CONFUSIÓN
# ==========================

cm = confusion_matrix(y_true, y_pred, labels=["Riesgo", "Exito"])

cm_df = pd.DataFrame(
    cm,
    index=["Real_Riesgo", "Real_Exito"],
    columns=["Pred_Riesgo", "Pred_Exito"]
)

cm_df.to_csv(os.path.join(RESULTS_DIR, "matriz_confusion_pycaret.csv"))

plt.figure()
plt.imshow(cm)
plt.title("Matriz de confusión - PyCaret")
plt.xticks([0, 1], ["Pred_Riesgo", "Pred_Exito"])
plt.yticks([0, 1], ["Real_Riesgo", "Real_Exito"])
plt.xlabel("Predicción")
plt.ylabel("Valor real")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "matriz_confusion_pycaret.png"))
plt.close()


# ==========================
# 9. GUARDAR MODELO
# ==========================

save_model(best_model, os.path.join(RESULTS_DIR, "modelo_pycaret"))


# ==========================
# 10. GUARDAR RESUMEN DE TIEMPOS
# ==========================

with open(os.path.join(RESULTS_DIR, "tiempos_pycaret.txt"), "w", encoding="utf-8") as f:
    f.write("TIEMPOS DE EJECUCIÓN - PROTOTIPO B PYCARET\n")
    f.write("==========================================\n")
    f.write(f"Tiempo de configuración setup: {setup_time:.4f} segundos\n")
    f.write(f"Tiempo de entrenamiento/comparación: {training_time:.4f} segundos\n")
    f.write(f"Tiempo de predicción: {prediction_time:.4f} segundos\n")
    f.write(f"Tiempo total: {setup_time + training_time + prediction_time:.4f} segundos\n")


print("Ejecución completada.")
print(f"Resultados guardados en: {RESULTS_DIR}")