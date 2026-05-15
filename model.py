"""
model.py
========
Red neuronal superficial para clasificación spam / no-spam.

Decisiones anti-overfitting:

- Arquitectura PEQUEÑA (16-8-1): el dataset tiene ~2.000 filas y sólo 5
  features. Redes más grandes memorizarían el train. La capacidad del
  modelo se mantiene baja a propósito.
- Regularización L2 en todas las capas densas.
- Dropout 0.3 entre capas para forzar redundancia.
- EarlyStopping monitorizando val_loss con paciencia 15.
- ReduceLROnPlateau para afinar al final del entrenamiento.
- class_weight automático por si el balance se rompe en algún fold.
- Validación K-FOLD ESTRATIFICADA (5 folds) para reportar un rendimiento
  honesto: el documento original sólo hablaba de un split simple, lo que
  da estimaciones muy variables con ~2.000 muestras.
- Métrica de selección: F1 sobre validación, no accuracy. El dataset
  está casi balanceado pero F1 es más conservador frente a overfitting
  hacia la clase mayoritaria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, regularizers


# ----------------------------------------------------------------------
# Construcción del modelo
# ----------------------------------------------------------------------

def build_model(n_features: int, l2: float = 1e-3, dropout: float = 0.3) -> tf.keras.Model:
    """Red superficial: Dense(16)-Dropout-Dense(8)-Dropout-Dense(1).

    Pequeña a propósito: 5 features y ~2k muestras no piden más capacidad.
    """
    inputs = layers.Input(shape=(n_features,), name="features")
    x = layers.Dense(
        16, activation="relu",
        kernel_regularizer=regularizers.l2(l2),
        name="hidden_1",
    )(inputs)
    x = layers.Dropout(dropout, name="dropout_1")(x)
    x = layers.Dense(
        8, activation="relu",
        kernel_regularizer=regularizers.l2(l2),
        name="hidden_2",
    )(x)
    x = layers.Dropout(dropout, name="dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="spam_prob")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="spam_detector")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def default_callbacks() -> List[callbacks.Callback]:
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=0,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-5,
            verbose=0,
        ),
    ]


# ----------------------------------------------------------------------
# Entrenamiento
# ----------------------------------------------------------------------

@dataclass
class TrainResult:
    model: tf.keras.Model
    history: dict
    val_metrics: dict
    test_metrics: dict = field(default_factory=dict)


def train_once(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *,
    epochs: int = 200,
    batch_size: int = 32,
    verbose: int = 0,
) -> TrainResult:
    """Entrena un único modelo con early-stopping sobre val."""
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    model = build_model(n_features=X_train.shape[1])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=default_callbacks(),
        class_weight=class_weight,
        verbose=verbose,
    )

    val_metrics = evaluate(model, X_val, y_val)
    return TrainResult(model=model, history=history.history, val_metrics=val_metrics)


def evaluate(model: tf.keras.Model, X: np.ndarray, y: np.ndarray) -> dict:
    probs = model.predict(X, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy":  float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall":    float(recall_score(y, preds, zero_division=0)),
        "f1":        float(f1_score(y, preds, zero_division=0)),
        "auc":       float(roc_auc_score(y, probs)),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
        "report":    classification_report(y, preds, zero_division=0, output_dict=True),
    }


def cross_validate(
    X: np.ndarray, y: np.ndarray,
    *,
    n_splits: int = 5,
    epochs: int = 200,
    batch_size: int = 32,
    random_state: int = 42,
) -> dict:
    """Validación K-Fold estratificada para estimar rendimiento de forma
    honesta. Devuelve métricas medias ± std a través de los folds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        result = train_once(X_tr, y_tr, X_va, y_va,
                            epochs=epochs, batch_size=batch_size)
        result.val_metrics["fold"] = fold
        fold_metrics.append(result.val_metrics)
        tf.keras.backend.clear_session()

    # Agregamos
    summary = {}
    for key in ("accuracy", "precision", "recall", "f1", "auc"):
        values = np.array([m[key] for m in fold_metrics])
        summary[key] = {"mean": float(values.mean()), "std": float(values.std())}
    summary["folds"] = fold_metrics
    return summary
