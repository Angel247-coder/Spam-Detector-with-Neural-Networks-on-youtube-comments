import os
from itertools import islice

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline

from transformers import pipeline
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
from wordcloud import WordCloud, STOPWORDS


# =========================================================
# 1) CONFIGURACIÓN
# =========================================================
st.set_page_config(page_title="IA YouTube Global Auditor", page_icon="🎬", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; color: #ff4b4b; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #eee; }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 2) UTILIDADES DE DATOS
# =========================================================
def detectar_columnas(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}

    texto_col = None
    etiqueta_col = None

    for candidato in ["content", "comentario", "text", "comment"]:
        if candidato in cols:
            texto_col = cols[candidato]
            break

    for candidato in ["class", "spam", "label", "etiqueta"]:
        if candidato in cols:
            etiqueta_col = cols[candidato]
            break

    return texto_col, etiqueta_col


def normalizar_etiquetas_spam(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int)

    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1,
        "spam": 1,
        "🚨 sí": 1,
        "si": 1,
        "sí": 1,
        "yes": 1,
        "true": 1,
        "0": 0,
        "no": 0,
        "✅ no": 0,
        "real": 0,
        "false": 0,
    }
    return s.map(mapping).fillna(0).astype(int)


@st.cache_data(show_spinner=False)
def cargar_dataset():
    posibles = [
        "Youtube_Unificado_Procesado.csv",
        "2026-05-13T17-45_export.csv",
    ]
    archivo = next((p for p in posibles if os.path.exists(p)), None)
    if archivo is None:
        raise FileNotFoundError(
            "No se encontró ni 'Youtube_Unificado_Procesado.csv' ni el CSV exportado cargado."
        )

    df = pd.read_csv(archivo)

    texto_col, etiqueta_col = detectar_columnas(df)
    if texto_col is None or etiqueta_col is None:
        raise ValueError(
            f"El CSV debe tener una columna de texto y una de etiqueta. Columnas encontradas: {list(df.columns)}"
        )

    df = df.copy()
    df = df.dropna(subset=[texto_col, etiqueta_col])
    df[texto_col] = df[texto_col].astype(str).str.strip()
    df = df[df[texto_col] != ""].copy()

    df["_texto"] = df[texto_col]
    df["_spam"] = normalizar_etiquetas_spam(df[etiqueta_col])

    return df, texto_col, etiqueta_col


# =========================================================
# 3) MODELOS
# =========================================================
@st.cache_resource(show_spinner=False)
def cargar_modelos(df: pd.DataFrame):
    X = df["_texto"]
    y = df["_spam"]

    # Mucho más robusto que una MLP para texto escaso:
    # - regularización L2
    # - balanceo de clases
    # - n-gramas para capturar patrones de spam
    modelo_spam = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words=None,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=6000,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            C=1.0,
            solver="liblinear",
            random_state=42
        ))
    ])

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    modelo_spam.fit(X_train, y_train)
    pred_val = modelo_spam.predict(X_val)

    metricas = {
        "accuracy": accuracy_score(y_val, pred_val),
        "f1": f1_score(y_val, pred_val, zero_division=0),
        "reporte": classification_report(y_val, pred_val, zero_division=0),
    }

    modelo_sentimiento = pipeline(
        "sentiment-analysis",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )

    return modelo_spam, modelo_sentimiento, metricas


def etiqueta_sentimiento(label: str) -> str:
    s = str(label).strip().lower()
    if "positive" in s or "pos" in s or "5 star" in s:
        return "Positive"
    if "negative" in s or "neg" in s or "1 star" in s:
        return "Negative"
    if "neutral" in s or "3 star" in s:
        return "Neutral"
    return str(label).capitalize()


def analizar_texto(texto: str, spam_model, sent_model):
    texto = "" if texto is None else str(texto).strip()

    if not texto:
        return {
            "spam": 0,
            "spam_conf": 0.0,
            "sentimiento": "Neutral",
            "sent_conf": 0.0,
        }

    probas = spam_model.predict_proba([texto])[0]
    pred = int(spam_model.predict([texto])[0])

    clases = spam_model.named_steps["clf"].classes_
    if 1 in clases:
        idx_spam = int(np.where(clases == 1)[0][0])
    else:
        idx_spam = int(np.argmax(probas))

    conf = float(probas[idx_spam]) * 100.0

    # Truncado para evitar errores con textos largos
    sent_raw = sent_model(texto[:1024], truncation=True)[0]
    sentimiento = etiqueta_sentimiento(sent_raw.get("label", "Neutral"))
    sent_conf = float(sent_raw.get("score", 0.0)) * 100.0

    return {
        "spam": pred,
        "spam_conf": conf,
        "sentimiento": sentimiento,
        "sent_conf": sent_conf,
    }


def extraer_comentarios(url: str, limite: int):
    downloader = YoutubeCommentDownloader()
    comentarios = islice(
        downloader.get_comments_from_url(url, sort_by=SORT_BY_RECENT),
        limite
    )

    salida = []
    for c in comentarios:
        texto = (c.get("text") or c.get("content") or "").strip()
        if not texto:
            continue

        salida.append({
            "Autor": c.get("author", "Desconocido"),
            "Comentario": texto,
            "Fecha": c.get("time", c.get("published_time", "")),
        })

    return salida


# =========================================================
# 4) APP
# =========================================================
def main():
    st.title("🎬 IA YouTube Global Auditor")

    try:
        df, texto_col, etiqueta_col = cargar_dataset()
        spam_model, sent_model, metricas = cargar_modelos(df)
    except Exception as e:
        st.error(f"No se pudo inicializar la app: {e}")
        st.stop()

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=50)
        st.title("YouTube IA Auditor")
        opcion = st.radio(
            "Menú Principal",
            ["Auditoría Video Real", "Análisis del Dataset", "Prueba Manual (Debug)"]
        )
        st.divider()
        if opcion == "Auditoría Video Real":
            num_c = st.slider("Comentarios a extraer", 10, 150, 50)
            st.caption("Más comentarios = más tiempo de proceso.")

    st.sidebar.success(
        f"Validación spam → Accuracy: {metricas['accuracy']:.3f} | F1: {metricas['f1']:.3f}"
    )

    if opcion == "Auditoría Video Real":
        st.header("🎬 Auditoría de Video en Tiempo Real")
        url = st.text_input("Pega la URL del video:", placeholder="https://www.youtube.com/watch?v=...")

        if st.button("🚀 Iniciar Análisis Profundo", type="primary"):
            if not url.strip():
                st.warning("Introduce una URL.")
            else:
                with st.spinner("Conectando con YouTube y analizando comentarios..."):
                    try:
                        comentarios = extraer_comentarios(url, num_c)

                        if not comentarios:
                            st.error("No se pudieron obtener comentarios o están desactivados.")
                            st.stop()

                        res = []
                        for c in comentarios:
                            salida = analizar_texto(c["Comentario"], spam_model, sent_model)
                            res.append({
                                "Autor": c["Autor"],
                                "Comentario": c["Comentario"],
                                "Tipo": "🚨 Spam" if salida["spam"] == 1 else "✅ Real",
                                "Sentimiento": salida["sentimiento"],
                                "Confianza Spam": f"{salida['spam_conf']:.1f}%",
                                "Confianza Sentimiento": f"{salida['sent_conf']:.1f}%",
                            })

                        df_res = pd.DataFrame(res)
                        if df_res.empty:
                            st.warning("No hay comentarios válidos para analizar.")
                            st.stop()

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Analizados", len(df_res))
                        c2.metric("Nivel de Spam", f"{(df_res['Tipo'] == '🚨 Spam').mean() * 100:.1f}%")
                        c3.metric("Felicidad Audiencia", f"{(df_res['Sentimiento'] == 'Positive').mean() * 100:.1f}%")

                        st.divider()

                        g1, g2 = st.columns(2)
                        with g1:
                            st.plotly_chart(
                                px.pie(
                                    df_res,
                                    names="Sentimiento",
                                    title="Clima Emocional",
                                    hole=0.35,
                                ),
                                use_container_width=True
                            )

                        with g2:
                            st.write("**Temas Reales (Nube de palabras)**")
                            txt_r = " ".join(df_res.loc[df_res["Tipo"] == "✅ Real", "Comentario"].astype(str))
                            if txt_r.strip():
                                wc = WordCloud(
                                    background_color="white",
                                    collocations=False,
                                    stopwords=set(STOPWORDS)
                                ).generate(txt_r)

                                fig, ax = plt.subplots(figsize=(8, 4))
                                ax.imshow(wc, interpolation="bilinear")
                                ax.axis("off")
                                st.pyplot(fig, clear_figure=True)
                            else:
                                st.info("No hay comentarios reales suficientes para generar la nube.")

                        st.subheader("📋 Auditoría Detallada")
                        st.dataframe(df_res, use_container_width=True)

                        st.download_button(
                            "📥 Descargar Reporte CSV",
                            df_res.to_csv(index=False).encode("utf-8"),
                            "auditoria.csv",
                            "text/csv"
                        )

                    except Exception as e:
                        st.error(f"Fallo en la conexión o en el análisis: {e}")

    elif opcion == "Análisis del Dataset":
        st.header("📊 Exploración del Dataset")
        st.write(f"Fuente detectada: `{texto_col}` / `{etiqueta_col}`")

        df_view = df.copy()
        df_view["Etiqueta"] = df_view["_spam"].map({0: "Real", 1: "Spam"})

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                px.pie(df_view, names="Etiqueta", title="Balance Spam/Real", hole=0.4),
                use_container_width=True
            )

        with col2:
            st.write("**Top Palabras en Spam**")
            spam_df = df_view[df_view["_spam"] == 1]

            if len(spam_df) < 2:
                st.info("No hay suficientes ejemplos de spam para extraer palabras clave.")
            else:
                cv = CountVectorizer(stop_words=None, max_features=10, ngram_range=(1, 1))
                cnts = cv.fit_transform(spam_df["_texto"])
                w_df = (
                    pd.DataFrame({
                        "Palabra": cv.get_feature_names_out(),
                        "Frecuencia": cnts.sum(axis=0).A1
                    })
                    .sort_values("Frecuencia", ascending=True)
                )
                st.plotly_chart(
                    px.bar(w_df, x="Frecuencia", y="Palabra", orientation="h"),
                    use_container_width=True
                )

        st.caption(
            f"Muestras: {len(df_view)} | Spam: {(df_view['_spam'] == 1).sum()} | Real: {(df_view['_spam'] == 0).sum()}"
        )

    else:
        st.header("🕵️ Prueba Manual de Comentarios")
        test_txt = st.text_area("Escribe algo para probar la IA:")

        if st.button("Analizar"):
            salida = analizar_texto(test_txt, spam_model, sent_model)
            c1, c2 = st.columns(2)
            c1.metric(
                "Detección",
                "🚨 SPAM" if salida["spam"] == 1 else "✅ REAL",
                f"{salida['spam_conf']:.1f}% confianza"
            )
            c2.metric(
                "Sentimiento",
                salida["sentimiento"],
                f"{salida['sent_conf']:.1f}% confianza"
            )

    st.divider()
    st.caption("Versión corregida | Menos overfitting + validación + funciones robustas")


if __name__ == "__main__":
    main()
