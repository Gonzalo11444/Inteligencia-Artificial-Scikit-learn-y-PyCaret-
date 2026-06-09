# Prototipo A – Scikit-learn

**TG3 – Desarrollo con Tecnologías Emergentes 2025-26 | Grupo 02**

## Descripción

Este prototipo implementa un modelo de clasificación binaria para predecir si un estudiante se encuentra en situación de **éxito académico** (`Exito`) o **riesgo académico** (`Riesgo`), usando el dataset educativo *Predict Students' Dropout and Academic Success*.

La tecnología empleada es **Scikit-learn**, la biblioteca de aprendizaje automático más utilizada en Python. A diferencia del prototipo B (PyCaret), aquí el pipeline completo —preprocesamiento, entrenamiento y evaluación— se construye de forma explícita, lo que ofrece mayor control sobre cada paso.

## Estructura de la carpeta

```
prototipo-A/
├── README.md
├── requirements.txt
├── data/
│   └── data.csv              ← Dataset educativo (separador ";")
├── src/
│   └── prototipo_sklearn.py  ← Script principal
└── results/                  ← Generado automáticamente al ejecutar
    ├── dataset_info_sklearn.txt
    ├── metricas_sklearn.csv
    ├── matriz_confusion_sklearn.csv
    ├── matriz_confusion_sklearn.png
    ├── classification_report_sklearn.txt
    ├── tiempos_sklearn.txt
    └── modelo_sklearn.pkl
```

## Requisitos previos

- Python 3.9 o superior (compatible con 3.10, 3.11, 3.12)
- El archivo `data/data.csv` con el dataset (separador de columnas: `;`)

## Instalación y ejecución

Desde la carpeta `prototipo-A/`:

```bash
# 1. Crear entorno virtual
python -m venv .venv

# 2. Activar el entorno
#    Windows:
.venv\Scripts\activate
#    Linux/macOS:
source .venv/bin/activate

# 3. Actualizar pip e instalar dependencias
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Ejecutar el prototipo
python src/prototipo_sklearn.py
```

## Parámetros configurables (Req. 15)

Al inicio de `prototipo_sklearn.py` se encuentran las constantes de configuración:

| Constante      | Valor por defecto       | Descripción                        |
|----------------|-------------------------|------------------------------------|
| `RANDOM_STATE` | `42`                    | Semilla para reproducibilidad      |
| `DATA_PATH`    | `data/data.csv`         | Ruta al dataset                    |
| `TEST_SIZE`    | `0.2`                   | Fracción de datos para prueba      |
| `TARGET_COL`   | `Target`                | Nombre de la columna objetivo      |

Para cambiar el modelo basta sustituir `LogisticRegression(...)` por otro estimador de Scikit-learn (p. ej. `RandomForestClassifier`, `SVC`, etc.) en la definición del `Pipeline`.

## Funcionamiento del prototipo

1. Carga el dataset desde `data/data.csv` (separador `;`).
2. Limpia los nombres de columnas (espacios y caracteres especiales).
3. Filtra solo los registros `Graduate` y `Dropout`; excluye `Enrolled`.
4. Crea la variable objetivo binaria `Target_binary` (`Exito` / `Riesgo`).
5. Separa variables predictoras (`X`) y variable objetivo (`y`).
6. Construye un `ColumnTransformer` con `StandardScaler` para variables numéricas y `OneHotEncoder` para categóricas.
7. Divide los datos en entrenamiento (80 %) y prueba (20 %) con semilla fija.
8. Entrena un `Pipeline` de Scikit-learn (preprocesador + `LogisticRegression`).
9. Mide el tiempo de entrenamiento y el tiempo de predicción por separado.
10. Calcula accuracy, precision, recall y F1-score sobre la clase `Riesgo`.
11. Genera y guarda la matriz de confusión (CSV y PNG).
12. Almacena todos los resultados y el modelo serializado en `results/`.

## Resultados generados

| Archivo                              | Contenido                                           |
|--------------------------------------|-----------------------------------------------------|
| `dataset_info_sklearn.txt`           | Tamaño del dataset y distribución de clases         |
| `metricas_sklearn.csv`               | Accuracy, Precision, Recall y F1-Score              |
| `matriz_confusion_sklearn.csv`       | Matriz de confusión en formato CSV                  |
| `matriz_confusion_sklearn.png`       | Imagen de la matriz de confusión                    |
| `classification_report_sklearn.txt`  | Informe completo por clase                          |
| `tiempos_sklearn.txt`                | Tiempos de entrenamiento, predicción y total        |
| `modelo_sklearn.pkl`                 | Modelo entrenado serializado con joblib             |

## Reproducibilidad (Req. 12)

La semilla `RANDOM_STATE = 42` se aplica en `train_test_split` y en `LogisticRegression`, garantizando resultados idénticos en ejecuciones consecutivas.

## Dependencias (Req. 14)

Listadas en `requirements.txt`: scikit-learn, pandas, numpy, matplotlib, joblib.
Compatible con Python ≥ 3.9, incluyendo Python 3.12.
