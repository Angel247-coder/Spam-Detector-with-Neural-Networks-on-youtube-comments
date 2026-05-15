"""
IA YouTube Global Auditor — v7.0 (Red Neuronal Superficial)

Cambios legales y técnicos respecto a versiones anteriores:
- Integración oficial de YouTube Data API v3.
- Cumplimiento estricto del RGPD (Seudonimización SHA-256).
- Reemplazo de Regresión Logística por Red Neuronal Superficial (Keras).
- Preprocesamiento ajustado: Eliminación de 'longitud_palabras' (r=0.91)
  y uso de RobustScaler para el manejo de outliers extremos.
"""

import hashlib
import os
import re
from difflib import SequenceMatcher

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from googleapiclient.discovery import build

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import RobustScaler
from transformers import pipeline as hf_pipeline
from wordcloud import STOPWORDS, WordCloud

# Importaciones para la Red Neuronal Superficial
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

for _r in ("vader_lexicon",):
    try:
        nltk.download(_r, quiet=True)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="IA YouTube Auditor", page_icon="🎬", layout="wide")
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; color: #ff4b4b; }
    .stMetric { background-color: #fff; padding: 15px; border-radius: 10px; border: 1px solid #eee; }
    .rgpd-box { background: #f0f4ff; border-left: 4px solid #3b82f6; padding: 12px 16px;
                border-radius: 6px; font-size: 0.85rem; margin-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# REGEX GLOBALES Y LÉXICO
# ─────────────────────────────────────────────────────────────────
URL_RE        = re.compile(r"https?://\S+|bit\.ly\S*|www\.\S+", re.I)
EXCL_RE       = re.compile(r"!{2,}")
CAPS_WORD_RE  = re.compile(r"\b[A-Z]{4,}\b")
EMOJI_RE      = re.compile(r"[^\x00-\x7F]")
REPEAT_RE     = re.compile(r"(\b\w+\b)(?:\s+\1){2,}", re.I)
TIMESTAMP_RE  = re.compile(r"\b\d{1,2}:\d{2}\b")
MENTION_RE    = re.compile(r"@\w+")
REPEAT_CHR_RE = re.compile(r"(.)\1{3,}")

SPAM_LEXICON = {
    "subscribe", "suscribete", "suscríbete", "free", "gratis", "click",
    "win", "gana", "money", "dinero", "cash", "giveaway", "sorteo",
    "check out", "visit", "my channel", "mi canal", "promo", "discount",
    "descuento", "link in bio", "crypto", "bitcoin", "investment",
    "earn", "profit", "dm me", "escríbeme",
}


# ─────────────────────────────────────────────────────────────────
# RGPD — SEUDONIMIZACIÓN (Art. 25)
# ─────────────────────────────────────────────────────────────────
def seudonimizar(nombre_real: str) -> str:
    digest = hashlib.sha256(nombre_real.encode("utf-8")).hexdigest()[:8].upper()
    return f"Usr-{digest}"


# ─────────────────────────────────────────────────────────────────
# EXTRACCIÓN DE VARIABLES PARA LA RED NEURONAL
# ─────────────────────────────────────────────────────────────────
class RedNeuronalFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._extraer_variables(t) for t in X], dtype=float)

    def _extraer_variables(self, text: str) -> list:
        t  = str(text)
        tl = t.lower()
        nc = max(len(t), 1)
        
        # 1. Variables del Sprint 2 adaptadas
        contiene_url = 1 if URL_RE.search(t) else 0
        ratio_mayusculas = sum(c.isupper() for c in t) / nc
        num_exclamaciones = len(EXCL_RE.findall(t))
        palabras_spam = sum(1 for w in SPAM_LEXICON if w in tl)
        longitud_caracteres = len(t)
        # OMITIDO: longitud_palabras (por colinealidad r=0.91 comprobada en el documento)

        # 2. Variables robustas añadidas
        emojis = len(EMOJI_RE.findall(t))
        palabras_caps = len(CAPS_WORD_RE.findall(t))
        repeticiones = len(REPEAT_RE.findall(tl))

        return [
            contiene_url,
            ratio_mayusculas,
            num_exclamaciones,
            palabras_spam,
            longitud_caracteres,
            emojis,
            palabras_caps,
            repeticiones
        ]


# ─────────────────────────────────────────────────────────────────
# UTILIDADES DE DATASET
# ─────────────────────────────────────────────────────────────────
def detectar_columnas(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    texto_col    = next((cols[k] for k in ("content", "comentario", "text", "comment") if k in cols), None)
    etiqueta_col = next((cols[k] for k in ("class", "spam", "label", "etiqueta") if k in cols), None)
    return texto_col, etiqueta_col

def normalizar_etiquetas(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int)
    mapa = {
        "1": 1, "spam": 1, "🚨 sí": 1, "si": 1, "sí": 1, "yes": 1, "true": 1,
        "0": 0, "no": 0,   "✅ no": 0, "real": 0, "false": 0,
    }
    return series.astype(str).str.strip().str.lower().map(mapa).fillna(0).astype(int)

@st.cache_data(show_spinner=False)
def cargar_dataset():
    posibles = ["Youtube_Unificado_Procesado.csv", "2026-05-13T17-45_export.csv", "Youtube01-Psy.csv"] # Añade tus nombres de archivo reales
    archivo  = next((p for p in posibles if os.path.exists(p)), None)
    if archivo is None:
        raise FileNotFoundError("No se encontró el CSV de entrenamiento en el directorio.")
    df = pd.read_csv(archivo)
    texto_col, etiqueta_col = detectar_columnas(df)
    if not texto_col or not etiqueta_col:
        raise ValueError(f"Columnas no reconocidas en el CSV.")
    df = df.dropna(subset=[texto_col, etiqueta_col]).copy()
    df["_texto"] = df[texto_col].astype(str).str.strip()
    df["_spam"]  = normalizar_etiquetas(df[etiqueta_col])
    return df[df["_texto"] != ""].copy(), texto_col, etiqueta_col


# ─────────────────────────────────────────────────────────────────
# MODELADO (PIPELINE + RED NEURONAL SUPERFICIAL)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def cargar_modelos(df: pd.DataFrame):
    X = df["_texto"].tolist()
    y = np.array(df["_spam"].tolist())

    # 1. Pipeline de preparación de datos
    pipeline_datos = Pipeline([
        ("features", FeatureUnion([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_df=0.95,
                max_features=2000, sublinear_tf=True, strip_accents="unicode",
            )),
            ("manual_features", RedNeuronalFeatures()),
        ])),
        # Usamos RobustScaler (with_centering=False para no romper la matriz dispersa de TFIDF)
        ("scaler", RobustScaler(with_centering=False))
    ])

    # Transformar textos a tensores numéricos
    X_tensor = pipeline_datos.fit_transform(X).toarray()
    
    # Split
    X_tr, X_val, y_tr, y_val = train_test_split(X_tensor, y, test_size=0.2, stratify=y, random_state=42)

    # 2. Arquitectura de la Red Neuronal Superficial
    input_dimension = X_tr.shape[1]
    modelo_rn = Sequential([
        Dense(64, activation='relu', input_shape=(input_dimension,)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    modelo_rn.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Entrenamiento rápido
    modelo_rn.fit(X_tr, y_tr, epochs=10, batch_size=32, validation_split=0.1, verbose=0)

    # 3. Evaluación
    y_pred_prob = modelo_rn.predict(X_val, verbose=0)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    metricas = {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1":       f1_score(y_val, y_pred, zero_division=0),
        "reporte":  classification_report(y_val, y_pred, zero_division=0),
    }

    # 4. Modelos de Sentimiento
    vader       = SentimentIntensityAnalyzer()
    transformer = hf_pipeline(
        "sentiment-analysis",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    )
    
    return pipeline_datos, modelo_rn, vader, transformer, metricas


# ─────────────────────────────────────────────────────────────────
# ToS YouTube — DESCARGA VÍA API OFICIAL
# ─────────────────────────────────────────────────────────────────
def extraer_video_id(url: str) -> str | None:
    patron = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([0-9A-Za-z_-]{11})", url)
    return patron.group(1) if patron else None

def descargar_comentarios_api(api_key: str, video_id: str, limite: int) -> list[dict]:
    youtube = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
    comentarios: list[dict] = []
    token_siguiente = None

    while len(comentarios) < limite:
        por_pagina = min(100, limite - len(comentarios))
        kwargs = dict(
            part="snippet",
            videoId=video_id,
            maxResults=por_pagina,
            order="time",
            textFormat="plainText",
        )
        if token_siguiente:
            kwargs["pageToken"] = token_siguiente

        respuesta = youtube.commentThreads().list(**kwargs).execute()

        for item in respuesta.get("items", []):
            snip        = item["snippet"]["topLevelComment"]["snippet"]
            nombre_real = snip.get("authorDisplayName", "")
            seudónimo   = seudonimizar(nombre_real)

            comentarios.append({
                "seudónimo": seudónimo,
                "texto":     snip.get("textDisplay", "").strip(),
            })

        token_siguiente = respuesta.get("nextPageToken")
        if not token_siguiente:
            break

    return [c for c in comentarios if c["texto"]][:limite]


# ─────────────────────────────────────────────────────────────────
# Detección y Análisis
# ─────────────────────────────────────────────────────────────────
def _similitud(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def detectar_spam_batch(comentarios: list[dict]) -> dict[int, bool]:
    resultado: dict[int, bool] = {i: False for i in range(len(comentarios))}
    por_autor: dict[str, list[int]] = {}

    for i, c in enumerate(comentarios):
        por_autor.setdefault(c["seudónimo"], []).append(i)

    for _, indices in por_autor.items():
        if len(indices) < 2: continue
        textos = [comentarios[i]["texto"] for i in indices]
        total = similares = 0
        for ia in range(len(textos)):
            for ib in range(ia + 1, len(textos)):
                total += 1
                if _similitud(textos[ia], textos[ib]) >= 0.80:
                    similares += 1
        if total > 0 and (similares / total) >= 0.60:
            for i in indices: resultado[i] = True

    return resultado

def reglas_duras(texto: str):
    t, tl, nc = str(texto), str(texto).lower(), max(len(str(texto)), 1)
    if URL_RE.search(t):                                               return True,  95.0
    if sum(c.isupper() for c in t) / nc > 0.60 and len(t) > 10:        return True,  90.0
    if EXCL_RE.search(t):                                              return True,  88.0
    if sum(1 for w in SPAM_LEXICON if w in tl) >= 3:                   return True,  87.0
    if REPEAT_RE.search(tl):                                           return True,  85.0
    if len(t.split()) <= 2:                                            return False, 80.0
    return None, None

def preprocesar_sent(texto: str) -> str:
    t = URL_RE.sub(" ", str(texto))
    t = TIMESTAMP_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = EMOJI_RE.sub(" ", t)
    t = REPEAT_CHR_RE.sub(r"\1\1", t)
    return re.sub(r"\s+", " ", t).strip()

def _vader_label(scores: dict) -> tuple[str, float]:
    c = scores["compound"]
    if c >=  0.05: return "Positive", min(0.5 + c / 2, 1.0) * 100
    if c <= -0.05: return "Negative", min(0.5 + abs(c) / 2, 1.0) * 100
    return "Neutral", (1.0 - abs(c)) * 100

def _transformer_label(raw: str) -> str:
    s = raw.strip().lower()
    if "positive" in s: return "Positive"
    if "negative" in s: return "Negative"
    return "Neutral"

def analizar_sentimiento(texto: str, vader, transformer) -> tuple[str, float]:
    limpio = preprocesar_sent(texto)
    if not limpio: return "Neutral", 50.0
    if len(limpio.split()) < 6: return _vader_label(vader.polarity_scores(limpio))
    
    res   = transformer(limpio[:512], truncation=True)[0]
    label = _transformer_label(res["label"])
    conf  = float(res["score"])
    
    if conf < 0.60:
        lv, _ = _vader_label(vader.polarity_scores(limpio))
        return (label, conf * 100) if label == lv else ("Neutral", 60.0)
    return label, conf * 100

def analizar(texto: str, pipeline_datos, modelo_rn, vader, transformer, batch_spam: bool = False) -> dict:
    texto = str(texto or "").strip()
    if not texto:
        return {"spam": 0, "spam_conf": 0.0, "sentimiento": "Neutral", "sent_conf": 50.0, "motivo": ""}

    if batch_spam:
        spam, spam_conf, motivo = 1, 97.0, "repetición (bot)"
    else:
        r_spam, r_conf = reglas_duras(texto)
        if r_spam is not None:
            spam, spam_conf, motivo = int(r_spam), r_conf, "regla" if r_spam else ""
        else:
            # 1. Transformar texto al tensor esperado por la red neuronal
            X_input = pipeline_datos.transform([texto]).toarray()
            
            # 2. Predicción con Keras
            proba_spam = float(modelo_rn.predict(X_input, verbose=0)[0][0])
            spam = 1 if proba_spam >= 0.5 else 0
            spam_conf = (proba_spam * 100) if spam else ((1 - proba_spam) * 100)
            motivo = "Red Neuronal" if spam else ""

    sentimiento, sent_conf = analizar_sentimiento(texto, vader, transformer)
    return {"spam": spam, "spam_conf": spam_conf,
            "sentimiento": sentimiento, "sent_conf": sent_conf, "motivo": motivo}


# ─────────────────────────────────────────────────────────────────
# AVISO DE PRIVACIDAD (RGPD Art. 13)
# ─────────────────────────────────────────────────────────────────
AVISO_PRIVACIDAD = """
**Aviso de privacidad — tratamiento de datos personales**

Esta herramienta accede a comentarios públicos de YouTube mediante la
**YouTube Data API v3**. Los nombres de usuario de los comentaristas son 
**datos personales** conforme al Art. 4.1 RGPD. El tratamiento se realiza bajo:

- **Seudonimización inmediata** (Art. 25 RGPD): el nombre real se convierte
  en un identificador irreversible (SHA-256) antes de procesarse.
- **Minimización** (Art. 5.1.c): sólo se tratan texto y seudónimo.
- **Limitación de conservación** (Art. 5.1.e): los datos existen únicamente
  durante la sesión activa de esta aplicación.
"""


# ─────────────────────────────────────────────────────────────────
# APP STREAMLIT
# ─────────────────────────────────────────────────────────────────
def main():
    st.title("🎬 IA YouTube Global Auditor (Red Neuronal)")

    try:
        df, texto_col, etiqueta_col = cargar_dataset()
    except Exception as e:
        st.error(str(e)); st.stop()

    with st.spinner("Entrenando Red Neuronal Superficial y cargando modelos…"):
        pipeline_datos, modelo_rn, vader, transformer, metricas = cargar_modelos(df)

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=50)
        st.title("YouTube IA Auditor")

        opcion = st.radio("Menú", [
            "Auditoría Video Real",
            "Análisis del Dataset",
            "Prueba Manual",
        ])
        st.divider()

        if opcion == "Auditoría Video Real":
            st.subheader("🔑 YouTube Data API v3")
            api_key = st.text_input(
                "API Key",
                type="password",
                help="Obtén tu clave en Google Cloud Console → APIs → YouTube Data API v3.",
            )
            num_c = st.slider("Comentarios a extraer", 10, 200, 50)
            st.caption("🔒 La API Key sólo se usa en esta sesión.")

        st.divider()
        st.success(f"Modelo RN — Acc {metricas['accuracy']:.2f} · F1 {metricas['f1']:.2f}")
        with st.expander("Reporte de validación (Keras)"):
            st.text(metricas["reporte"])

        st.divider()
        with st.expander("📋 Aviso de privacidad (RGPD)"):
            st.markdown(AVISO_PRIVACIDAD)

    # ── A) Auditoría real ─────────────────────────────────────────
    if opcion == "Auditoría Video Real":
        st.header("🎬 Auditoría en Tiempo Real")

        st.markdown(
            '<div class="rgpd-box">🔒 <strong>Privacidad:</strong> los nombres de usuario se seudonimizarán con SHA-256.</div>',
            unsafe_allow_html=True,
        )

        url = st.text_input("URL del vídeo:", placeholder="https://www.youtube.com/watch?v=...")

        if st.button("🚀 Analizar", type="primary"):
            if not url.strip():
                st.warning("Introduce una URL."); return
            if not api_key.strip():
                st.warning("Introduce tu API Key en el panel lateral."); return

            video_id = extraer_video_id(url)
            if not video_id:
                st.error("No se pudo extraer el ID del vídeo."); return

            with st.spinner("Descargando comentarios vía API oficial…"):
                try:
                    comentarios = descargar_comentarios_api(api_key, video_id, num_c)
                except Exception as e:
                    st.error(f"Error de la YouTube API: {e}")
                    return

            if not comentarios:
                st.error("No se obtuvieron comentarios."); return

            with st.spinner("Analizando con Red Neuronal…"):
                flags_batch = detectar_spam_batch(comentarios)

                filas = []
                for i, c in enumerate(comentarios):
                    res = analizar(
                        c["texto"], pipeline_datos, modelo_rn, vader, transformer,
                        batch_spam=flags_batch[i],
                    )
                    filas.append({
                        "Seudónimo":    c["seudónimo"],
                        "Comentario":   c["texto"],
                        "Spam":         "🚨 SÍ" if res["spam"] else "✅ NO",
                        "Motivo":       res["motivo"],
                        "Sentimiento":  res["sentimiento"],
                        "Conf. spam":   f"{res['spam_conf']:.0f}%",
                        "Conf. sent.":  f"{res['sent_conf']:.0f}%",
                    })

            df_res = pd.DataFrame(filas)

            c1, c2, c3 = st.columns(3)
            c1.metric("Analizados", len(df_res))
            c2.metric("Spam detectado", f"{(df_res['Spam']=='🚨 SÍ').mean()*100:.1f}%", delta_color="inverse")
            c3.metric("Audiencia positiva", f"{(df_res['Sentimiento']=='Positive').mean()*100:.1f}%")

            st.divider()
            g1, g2 = st.columns(2)

            with g1:
                st.plotly_chart(px.pie(
                    df_res, names="Sentimiento", title="Distribución de sentimiento", hole=0.35,
                    color="Sentimiento",
                    color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral":  "#95a5a6"},
                ), use_container_width=True)

            with g2:
                st.write("**Nube de palabras — comentarios reales**")
                txt = " ".join(df_res.loc[df_res["Spam"] == "✅ NO", "Comentario"].astype(str))
                if txt.strip():
                    wc = WordCloud(background_color="white", collocations=False, stopwords=set(STOPWORDS)).generate(txt)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
                    st.pyplot(fig, clear_figure=True)
                else:
                    st.info("No hay comentarios reales suficientes.")

            st.subheader("📋 Detalle completo")
            st.dataframe(df_res, use_container_width=True)

            csv_bytes = df_res.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Descargar CSV (datos anonimizados)", csv_bytes, "auditoria_anonimizada.csv", "text/csv")

    # ── B) Dataset ────────────────────────────────────────────────
    elif opcion == "Análisis del Dataset":
        st.header("📊 Dataset de entrenamiento")
        df_v = df.assign(Etiqueta=df["_spam"].map({0: "Real", 1: "Spam"}))

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.pie(
                df_v, names="Etiqueta", title="Balance Real / Spam", hole=0.4,
            ), use_container_width=True)
        with col2:
            spam_df = df_v[df_v["_spam"] == 1]
            if len(spam_df) >= 2:
                cv   = TfidfVectorizer(max_features=12, ngram_range=(1, 2))
                cnts = cv.fit_transform(spam_df["_texto"])
                w_df = pd.DataFrame({
                    "Término":    cv.get_feature_names_out(),
                    "Frecuencia": cnts.sum(axis=0).A1,
                }).sort_values("Frecuencia")
                st.plotly_chart(px.bar(
                    w_df, x="Frecuencia", y="Término", orientation="h", title="Top términos en spam (TF-IDF)",
                ), use_container_width=True)

        st.caption(f"{len(df_v)} muestras · {(df_v['_spam']==1).sum()} spam · {(df_v['_spam']==0).sum()} reales")

    # ── C) Prueba manual ──────────────────────────────────────────
    else:
        st.header("🕵️ Prueba manual")
        texto = st.text_area("Escribe un comentario de prueba:")

        if st.button("Analizar"):
            if not texto.strip():
                st.warning("Escribe algo primero."); return

            res = analizar(texto, pipeline_datos, modelo_rn, vader, transformer)
            c1, c2 = st.columns(2)
            c1.metric("Spam", "🚨 SÍ" if res["spam"] else "✅ NO", f"{res['spam_conf']:.0f}% confianza (Keras)")
            c2.metric("Sentimiento", res["sentimiento"], f"{res['sent_conf']:.0f}% confianza")

            with st.expander("🔍 Variables extraídas (Input Red Neuronal)"):
                vals    = RedNeuronalFeatures()._extraer_variables(texto)
                nombres = ["Contiene URL", "Ratio Mayúsculas", "Exclamaciones", "Palabras Spam", 
                           "Longitud Caracteres", "Emojis", "Palabras en CAPS", "Repeticiones"]
                for n, v in zip(nombres, vals):
                    st.write(f"**{n}**: {v:.3f}")

            with st.expander("🔍 VADER scores"):
                limpio = preprocesar_sent(texto)
                for k, v in vader.polarity_scores(limpio).items():
                    st.write(f"**{k}**: {v:.3f}")


if __name__ == "__main__":
    main()
