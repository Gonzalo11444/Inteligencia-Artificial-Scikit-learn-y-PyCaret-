# Prototipo B - PyCaret

## Descripción

Este prototipo utiliza la biblioteca PyCaret para desarrollar un modelo de clasificación aplicado al ámbito educativo. El objetivo es predecir si un estudiante se encuentra en situación de éxito académico o riesgo académico a partir de un dataset de estudiantes.

El dataset utilizado procede del conjunto "Predict Students' Dropout and Academic Success" de UCI Machine Learning Repository.

Para simplificar el problema y ajustarlo a los requisitos del TG3, se han utilizado únicamente los registros con valores `Graduate` y `Dropout` en la variable `Target`.

- `Graduate` se transforma en `Exito`
- `Dropout` se transforma en `Riesgo`
- Los registros `Enrolled` se excluyen del experimento

## Tecnología utilizada

- Python
- PyCaret
- Pandas
- Scikit-learn
- Matplotlib

## Estructura de la carpeta

```text
prototipo-B/
├── README.md
├── requirements.txt
├── data/
│   └── data.csv
├── src/
│   └── prototipo_pycaret.py
└── results/

## Instalación

Se recomienda usar Python 3.10 o 3.11, ya que PyCaret 3.3.2 no es compatible con Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt