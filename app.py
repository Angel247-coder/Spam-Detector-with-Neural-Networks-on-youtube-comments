"""
app.py
======
Aplicación Streamlit del detector de spam.

Modos de entrada:
    1. Texto individual
    2. Archivo CSV con columna CONTENT
    3. URL de vídeo de YouTube → descarga comentarios vía API y los
       clasifica. (Esta opción amplía el alcance del documento original,
       que excluía explícitamente la conexión a la API.)

Ejecutar:
    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from data_prep import FEATURE_COLUMNS, featurize_raw_text
from youtube_api import fetch_comments


MODEL_PATH  = Path("artifacts/spam_model.keras")
SCALER_PATH = Path("artifacts/scaler.joblib")


# ----------------------------------------------------------------------
# Carga de artefactos (cacheada)
# ----------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        st.error(
            "No se encuentran los artefactos del modelo. "
            "Entrena primero con: `python train.py`"
        )
        st.stop()
    model  = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def classify(texts: list[str], model, scaler, threshold: float = 0.5) -> pd.DataFrame:
    X = featurize_raw_text(texts, scaler)
    probs = model.predict(X, verbose=0).ravel()
    preds = (probs >= threshold).astype(int)
    return pd.DataFrame({
        "comentario": texts,
        "prob_spam":  probs.round(4),
        "prediccion": np.where(preds == 1, "SPAM", "NO SPAM"),
    })


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Detector de Spam — YouTube", layout="wide")
    st.title("🛡️ Detector de Spam en comentarios de YouTube")
    st.caption(
        "Red neuronal superficial entrenada con 5 datasets de comentarios "
        "(Psy, Katy Perry, LMFAO, Eminem, Shakira)."
    )

    model, scaler = load_artifacts()

    with st.sidebar:
        st.header("Configuración")
        threshold = st.slider(
            "Umbral de decisión",
            min_value=0.10, max_value=0.90, value=0.50, step=0.05,
            help="Por encima del umbral, el comentario se considera spam.",
        )
        st.caption("Features usadas:")
        st.code("\n".join(FEATURE_COLUMNS), language="text")

    tab_text, tab_file, tab_yt = st.tabs(
        ["📝 Texto", "📁 Archivo CSV", "🎬 URL de YouTube"]
    )

    # --- Modo 1: texto individual ------------------------------------
    with tab_text:
        st.subheader("Clasificar un comentario")
        comment = st.text_area("Pega el comentario aquí", height=140)
        if st.button("Analizar", key="btn_text"):
            if not comment.strip():
                st.warning("Introduce un comentario.")
            else:
                df = classify([comment], model, scaler, threshold)
                row = df.iloc[0]
                if row["prediccion"] == "SPAM":
                    st.error(f"🚨 SPAM — probabilidad {row['prob_spam']:.2%}")
                else:
                    st.success(f"✅ NO SPAM — probabilidad de spam {row['prob_spam']:.2%}")

    # --- Modo 2: archivo CSV -----------------------------------------
    with tab_file:
        st.subheader("Clasificar un CSV de comentarios")
        st.caption("El CSV debe tener una columna llamada `CONTENT`.")
        upload = st.file_uploader("Sube el CSV", type=["csv"])
        if upload is not None:
            df_in = pd.read_csv(upload)
            if "CONTENT" not in df_in.columns:
                st.error("El CSV no contiene una columna `CONTENT`.")
            else:
                texts = df_in["CONTENT"].astype(str).tolist()
                results = classify(texts, model, scaler, threshold)
                st.dataframe(results, use_container_width=True)
                spam_count = (results["prediccion"] == "SPAM").sum()
                st.metric("Comentarios marcados como spam",
                          f"{spam_count} / {len(results)}")
                st.download_button(
                    "Descargar resultados (CSV)",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="clasificacion_spam.csv",
                    mime="text/csv",
                )

    # --- Modo 3: URL de YouTube --------------------------------------
    with tab_yt:
        st.subheader("Clasificar comentarios en directo desde YouTube")
        st.info(
            "Esta opción se conecta a la **YouTube Data API v3**. "
            "Necesitas una API key gratuita de Google Cloud Console."
        )
        default_key = os.environ.get("YOUTUBE_API_KEY", "")
        api_key = st.text_input(
            "API key de YouTube Data API v3",
            value=default_key, type="password",
        )
        url = st.text_input(
            "URL o ID del vídeo",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        max_comments = st.slider("Máximo de comentarios a descargar",
                                 50, 1000, 200, step=50)

        if st.button("Descargar y analizar", key="btn_yt"):
            if not api_key:
                st.warning("Introduce tu API key.")
            elif not url.strip():
                st.warning("Introduce la URL o ID del vídeo.")
            else:
                with st.spinner("Descargando comentarios..."):
                    try:
                        comments = fetch_comments(url, api_key, max_comments=max_comments)
                    except Exception as exc:
                        st.error(f"Error: {exc}")
                        return
                if not comments:
                    st.warning("El vídeo no tiene comentarios o están desactivados.")
                else:
                    texts = [c["content"] for c in comments]
                    results = classify(texts, model, scaler, threshold)
                    meta = pd.DataFrame(comments)[["author", "date"]]
                    full = pd.concat([meta, results], axis=1)
                    st.dataframe(full, use_container_width=True)
                    spam_count = (results["prediccion"] == "SPAM").sum()
                    st.metric("Spam detectado",
                              f"{spam_count} / {len(results)} "
                              f"({spam_count/len(results):.1%})")
                    st.download_button(
                        "Descargar resultados (CSV)",
                        data=full.to_csv(index=False).encode("utf-8"),
                        file_name="youtube_spam.csv",
                        mime="text/csv",
                    )


if __name__ == "__main__":
    main()
