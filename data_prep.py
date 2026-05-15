"""
data_prep.py
============
Carga los 5 CSV de YouTube, unifica, añade variables derivadas, trata
outliers y ausentes según las decisiones del documento de la memoria,
y devuelve X / y listos para entrenar.

Bugs corregidos respecto a los fragmentos del documento original:

1. La columna 'Artista' NO existe en los CSV originales — había que CREARLA
   al concatenar. El snippet 1.2 del documento la usaba como si existiera
   y habría lanzado KeyError. Aquí la añadimos en la unificación.

2. La regex r'http|www|.com' del snippet 1.3 tiene '.com' sin escapar,
   por lo que el punto matchea CUALQUIER carácter y produce falsos
   positivos (p.ej. "acomódate" contendría "acom"). Se escapa a r'\\.com\\b'
   y se ancla a límite de palabra.

3. ratio_mayusculas dividía por longitud_caracteres pero ese cálculo se
   hace ANTES de longitud_caracteres en el orden del documento ⇒ NaN.
   Aquí calculamos las longitudes primero.

4. cantidad_exclamaciones contaba '!' solo una vez por comentario en
   algunas implementaciones del equipo. Usamos str.count para contar TODAS.

5. El documento decide eliminar 'longitud_palabras' por colinealidad
   (r=0.91) — se respeta aquí en la matriz de features final.

6. Las normalizaciones del documento (scaler.fit_transform aplicado a
   todo el dataset) producen FUGA DE DATOS (data leakage). Aquí el
   scaler SOLO se ajusta sobre train; se aplica a val/test sin re-fit.

7. La columna DATE se descarta (no es feature predictiva y tiene 12.5%
   de ausentes según el doc) sin perder filas: el doc proponía eliminar
   245 filas, pero como DATE no entra al modelo, eliminarlas reduce
   datos de entrenamiento sin razón. Mantenemos las filas y descartamos
   la columna.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


# ----------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------

CSV_FILES = {
    "Psy":       "Youtube01-Psy.csv",
    "KatyPerry": "Youtube02-KatyPerry.csv",
    "LMFAO":     "Youtube03-LMFAO.csv",
    "Eminem":    "Youtube04-Eminem.csv",
    "Shakira":   "Youtube05-Shakira.csv",
}

# Patrón de URL más estricto que el del documento
URL_REGEX = re.compile(
    r"https?://\S+|www\.\S+|\b\S+\.(?:com|net|org|io|co|tv|me|ly|gl)\b",
    flags=re.IGNORECASE,
)

# Palabras gancho ampliadas. El documento usaba sólo
# 'check|channel|subscribe|my video|sub' — añadimos variantes comunes
# en spam de YouTube manteniendo la lista corta para no sobreajustar.
SPAM_KEYWORDS = re.compile(
    r"\b(?:check|channel|subscribe|sub|click|free|gift|win|prize|"
    r"money|earn|cash|please|plz|pls|my video|my channel|visit)\b",
    flags=re.IGNORECASE,
)

# Features finales que entran a la red. 'longitud_palabras' se descarta
# por colinealidad con 'longitud_caracteres' (r=0.91), según decisión
# del documento (sección 4.6).
FEATURE_COLUMNS = [
    "longitud_caracteres",
    "ratio_mayusculas",
    "cantidad_exclamaciones",
    "contiene_url",
    "contiene_palabras_spam",
]

TARGET_COLUMN = "CLASS"


# ----------------------------------------------------------------------
# Carga y unificación
# ----------------------------------------------------------------------

def load_and_unify(csv_dir: str | Path) -> pd.DataFrame:
    """Carga los 5 CSV y los une en un único DataFrame, añadiendo la
    columna 'Artista' que el documento usaba sin crearla."""
    csv_dir = Path(csv_dir)
    frames = []
    for artist, filename in CSV_FILES.items():
        path = csv_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"No se encuentra {path}. Coloca los 5 CSV en {csv_dir}."
            )
        df = pd.read_csv(path)
        df["Artista"] = artist          # ← BUG FIX: la columna no existía
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------
# Ingeniería de variables
# ----------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade las 6 variables derivadas del documento, en el orden
    correcto para que ratio_mayusculas no produzca NaN."""
    df = df.copy()
    df["CONTENT"] = df["CONTENT"].astype(str)

    # 1) Longitudes PRIMERO (ratio_mayusculas las necesita)
    df["longitud_caracteres"] = df["CONTENT"].str.len()
    df["longitud_palabras"]   = df["CONTENT"].str.split().str.len()

    # 2) Ratio de mayúsculas — protegido contra división por cero
    upper_count = df["CONTENT"].str.count(r"[A-Z]")
    df["ratio_mayusculas"] = np.where(
        df["longitud_caracteres"] > 0,
        upper_count / df["longitud_caracteres"],
        0.0,
    )

    # 3) Exclamaciones (cuenta TODAS, no sólo presencia)
    df["cantidad_exclamaciones"] = df["CONTENT"].str.count(r"!")

    # 4) Presencia de URL (regex corregida)
    df["contiene_url"] = df["CONTENT"].str.contains(URL_REGEX, na=False).astype(int)

    # 5) Presencia de palabras gancho de spam
    df["contiene_palabras_spam"] = (
        df["CONTENT"].str.contains(SPAM_KEYWORDS, na=False).astype(int)
    )

    return df


# ----------------------------------------------------------------------
# Limpieza y split
# ----------------------------------------------------------------------

def prepare_for_training(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, RobustScaler]:
    """Devuelve X_train, X_val, X_test, y_train, y_val, y_test, scaler.

    Importante: el RobustScaler se AJUSTA SOLO con train. Esto evita la
    fuga de datos del snippet del documento, donde el scaler se hacía
    fit_transform sobre todo el dataset antes del split.
    """
    # Aseguramos que CLASS sea entero y descartamos filas sin label
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(np.int32)

    # Split estratificado: 70/15/15
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=relative_val,
        random_state=random_state,
        stratify=y_temp,
    )

    # RobustScaler — fit SÓLO con train (evita data leakage)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def featurize_raw_text(texts: list[str], scaler: RobustScaler) -> np.ndarray:
    """Convierte una lista de comentarios crudos en la matriz X escalada
    lista para predecir. Usa el scaler ya entrenado."""
    df = pd.DataFrame({"CONTENT": texts})
    df = add_features(df)
    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    return scaler.transform(X).astype(np.float32)
