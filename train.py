"""
train.py
========
Punto de entrada para entrenar y guardar el modelo.

Flujo:
    1. Carga y unifica los 5 CSV.
    2. Añade features derivadas.
    3. Reporta métricas con K-Fold (5 folds) para una estimación honesta.
    4. Entrena el modelo final con split train/val/test 70/15/15.
    5. Guarda el modelo y el scaler en artifacts/.

Uso:
    python train.py --csv-dir ./data --epochs 200

Salidas:
    artifacts/spam_model.keras
    artifacts/scaler.joblib
    artifacts/metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from data_prep import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_features,
    load_and_unify,
    prepare_for_training,
)
from model import cross_validate, evaluate, train_once


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena el detector de spam.")
    p.add_argument("--csv-dir", default="./data",
                   help="Carpeta con los 5 CSV Youtube0X-*.csv")
    p.add_argument("--artifacts-dir", default="./artifacts")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--skip-cv", action="store_true",
                   help="Salta la validación cruzada (más rápido).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1) Datos
    print("→ Cargando y unificando CSVs...")
    df_raw = load_and_unify(args.csv_dir)
    print(f"   filas: {len(df_raw)}")

    print("→ Añadiendo features...")
    df = add_features(df_raw)
    print(f"   features: {FEATURE_COLUMNS}")

    # 2) K-Fold CV honesto sobre TODO el dataset (excepto el test final)
    if not args.skip_cv:
        print("→ Validación cruzada 5-fold...")
        # Para CV usamos los datos antes del scaling y escalamos dentro de cada fold
        from sklearn.preprocessing import RobustScaler

        X_all = df[FEATURE_COLUMNS].values.astype(np.float32)
        y_all = df[TARGET_COLUMN].astype(int).values

        # Escalamos por fold internamente — usamos un wrapper rápido
        # (re-escalar dentro de cada fold sería lo más correcto, pero por
        # simplicidad y dado que RobustScaler es muy estable, escalamos
        # con un scaler ajustado a los folds de train de forma agregada
        # NO usando el test final). Para honestidad estricta:
        from sklearn.model_selection import StratifiedKFold
        from model import train_once as _train_once

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        fold_metrics = []
        for fold, (tr, va) in enumerate(skf.split(X_all, y_all), start=1):
            sc = RobustScaler().fit(X_all[tr])
            Xtr, Xva = sc.transform(X_all[tr]).astype(np.float32), sc.transform(X_all[va]).astype(np.float32)
            ytr, yva = y_all[tr], y_all[va]
            res = _train_once(Xtr, ytr, Xva, yva,
                              epochs=args.epochs, batch_size=args.batch_size)
            res.val_metrics["fold"] = fold
            fold_metrics.append(res.val_metrics)
            print(f"   fold {fold}: acc={res.val_metrics['accuracy']:.4f} "
                  f"f1={res.val_metrics['f1']:.4f} auc={res.val_metrics['auc']:.4f}")
            tf.keras.backend.clear_session()

        cv_summary = {}
        for k in ("accuracy", "precision", "recall", "f1", "auc"):
            arr = np.array([m[k] for m in fold_metrics])
            cv_summary[k] = {"mean": float(arr.mean()), "std": float(arr.std())}
        print("   CV resumen:")
        for k, v in cv_summary.items():
            print(f"     {k:10s}: {v['mean']:.4f} ± {v['std']:.4f}")
    else:
        cv_summary = None

    # 3) Modelo final
    print("→ Entrenando modelo final (train/val/test = 70/15/15)...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_for_training(
        df, test_size=0.15, val_size=0.15, random_state=args.seed,
    )
    result = train_once(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs, batch_size=args.batch_size, verbose=2,
    )
    test_metrics = evaluate(result.model, X_test, y_test)
    print("→ Métricas en TEST:")
    for k in ("accuracy", "precision", "recall", "f1", "auc"):
        print(f"     {k:10s}: {test_metrics[k]:.4f}")
    print(f"   matriz de confusión: {test_metrics['confusion_matrix']}")

    # 4) Guardar artefactos
    model_path  = artifacts_dir / "spam_model.keras"
    scaler_path = artifacts_dir / "scaler.joblib"
    metrics_path = artifacts_dir / "metrics.json"

    result.model.save(model_path)
    joblib.dump(scaler, scaler_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cv_summary": cv_summary,
                "val_metrics":  {k: v for k, v in result.val_metrics.items()
                                 if k != "report"},
                "test_metrics": {k: v for k, v in test_metrics.items()
                                 if k != "report"},
                "features": FEATURE_COLUMNS,
            },
            f, indent=2, ensure_ascii=False,
        )

    print(f"\n✓ Modelo guardado en {model_path}")
    print(f"✓ Scaler guardado en {scaler_path}")
    print(f"✓ Métricas en {metrics_path}")


if __name__ == "__main__":
    main()
