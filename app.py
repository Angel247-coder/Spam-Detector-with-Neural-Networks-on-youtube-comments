
import os
import re
from difflib import SequenceMatcher
from itertools import islice

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from transformers import pipeline as hf_pipeline
from wordcloud import STOPWORDS, WordCloud
from youtube_comment_downloader import SORT_BY_RECENT, YoutubeCommentDownloader

for _r in ("vader_lexicon",):
    try:
        nltk.download(_r, quiet=True)
    except Exception:
        pass

from nltk.sentiment.vader import SentimentIntensityAnalyzer


# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="IA YouTube Auditor", page_icon="🎬", layout="wide")
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; color: #ff4b4b; }
    .stMetric { background-color: #fff; padding: 15px; border-radius: 10px; border: 1px solid #eee; }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# REGEX GLOBALES
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
# FEATURES MANUALES (sklearn Transformer)
# ─────────────────────────────────────────────────────────────────
class HandcraftedSpamFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._f(t) for t in X], dtype=float)

    def _f(self, text: str) -> list:
        t  = str(text)
        tl = t.lower()
        ws = tl.split()
        nc = max(len(t), 1)
        nw = max(len(ws), 1)
        return [
            len(URL_RE.findall(t)),
            sum(c.isupper() for c in t) / nc,
            t.count("!"),
            len(EXCL_RE.findall(t)),
            len(EMOJI_RE.findall(t)),
            len(CAPS_WORD_RE.findall(t)),
            len(set(ws)) / nw,
            sum(1 for w in SPAM_LEXICON if w in tl),
            len(REPEAT_RE.findall(tl)),
            len(t),
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
    posibles = ["Youtube_Unificado_Procesado.csv", "2026-05-13T17-45_export.csv"]
    archivo  = next((p for p in posibles if os.path.exists(p)), None)
    if archivo is None:
        raise FileNotFoundError("No se encontró el CSV de entrenamiento.")
    df = pd.read_csv(archivo)
    texto_col, etiqueta_col = detectar_columnas(df)
    if not texto_col or not etiqueta_col:
        raise ValueError(f"Columnas no reconocidas: {list(df.columns)}")
    df = df.dropna(subset=[texto_col, etiqueta_col]).copy()
    df["_texto"] = df[texto_col].astype(str).str.strip()
    df["_spam"]  = normalizar_etiquetas(df[etiqueta_col])
    return df[df["_texto"] != ""].copy(), texto_col, etiqueta_col


# ─────────────────────────────────────────────────────────────────
# MODELOS
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def cargar_modelos(df: pd.DataFrame):
    X = df["_texto"].tolist()
    y = df["_spam"].tolist()

    spam_pipe = Pipeline([
        ("features", FeatureUnion([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_df=0.95,
                max_features=5000, sublinear_tf=True, strip_accents="unicode",
            )),
            ("hc", HandcraftedSpamFeatures()),
        ])),
        ("scaler", MaxAbsScaler()),
        ("clf", LogisticRegression(
            C=0.3, class_weight="balanced",
            solver="liblinear", max_iter=2000, random_state=42,
        )),
    ])

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    spam_pipe.fit(X_tr, y_tr)
    y_pred = spam_pipe.predict(X_val)

    metricas = {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1":       f1_score(y_val, y_pred, zero_division=0),
        "reporte":  classification_report(y_val, y_pred, zero_division=0),
    }

    vader      = SentimentIntensityAnalyzer()
    transformer = hf_pipeline(
        "sentiment-analysis",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    )
    return spam_pipe, vader, transformer, metricas


# ─────────────────────────────────────────────────────────────────
# SPAM — CAPA 1: DUPLICADOS EN BATCH
# ─────────────────────────────────────────────────────────────────
def _similitud(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def detectar_spam_batch(comentarios: list[dict]) -> dict[int, bool]:
    """
    Marca como spam los comentarios de autores que repiten texto muy similar
    (ratio ≥ 0.80) más de una vez dentro del batch analizado.
    """
    resultado: dict[int, bool] = {i: False for i in range(len(comentarios))}
    por_autor: dict[str, list[int]] = {}

    for i, c in enumerate(comentarios):
        autor = c.get("Autor", "").strip().lower()
        por_autor.setdefault(autor, []).append(i)

    for autor, indices in por_autor.items():
        if len(indices) < 2:
            continue
        textos = [comentarios[i]["Comentario"] for i in indices]
        total = similares = 0
        for ia in range(len(textos)):
            for ib in range(ia + 1, len(textos)):
                total += 1
                if _similitud(textos[ia], textos[ib]) >= 0.80:
                    similares += 1
        if total > 0 and (similares / total) >= 0.60:
            for i in indices:
                resultado[i] = True

    return resultado


# ─────────────────────────────────────────────────────────────────
# SPAM — CAPA 2: REGLAS DURAS
# ─────────────────────────────────────────────────────────────────
def reglas_duras(texto: str):
    """Devuelve (spam: bool, confianza: float) o (None, None) si no concluyente."""
    t  = str(texto)
    tl = t.lower()
    nc = max(len(t), 1)

    if URL_RE.search(t):
        return True, 95.0
    if sum(c.isupper() for c in t) / nc > 0.60 and len(t) > 10:
        return True, 90.0
    if EXCL_RE.search(t):
        return True, 88.0
    if sum(1 for w in SPAM_LEXICON if w in tl) >= 3:
        return True, 87.0
    if REPEAT_RE.search(tl):
        return True, 85.0
    if len(t.split()) <= 2:           # timestamp, "lol", "7:12" → casi nunca spam
        return False, 80.0
    return None, None


# ─────────────────────────────────────────────────────────────────
# SENTIMIENTO
# ─────────────────────────────────────────────────────────────────
def preprocesar_sent(texto: str) -> str:
    t = URL_RE.sub(" ", str(texto))
    t = TIMESTAMP_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = EMOJI_RE.sub(" ", t)
    t = REPEAT_CHR_RE.sub(r"\1\1", t)
    return re.sub(r"\s+", " ", t).strip()


def _vader_label(scores: dict) -> tuple[str, float]:
    c = scores["compound"]
    if c >= 0.05:  return "Positive", min(0.5 + c / 2, 1.0) * 100
    if c <= -0.05: return "Negative", min(0.5 + abs(c) / 2, 1.0) * 100
    return "Neutral", (1.0 - abs(c)) * 100


def _transformer_label(raw: str) -> str:
    s = raw.strip().lower()
    if "positive" in s: return "Positive"
    if "negative" in s: return "Negative"
    return "Neutral"


def analizar_sentimiento(texto: str, vader, transformer) -> tuple[str, float]:
    limpio = preprocesar_sent(texto)
    if not limpio:
        return "Neutral", 50.0

    palabras = limpio.split()

    # Textos cortos → VADER es más fiable
    if len(palabras) < 6:
        return _vader_label(vader.polarity_scores(limpio))

    # Transformer para textos con contexto
    res    = transformer(limpio[:512], truncation=True)[0]
    label  = _transformer_label(res["label"])
    conf   = float(res["score"])

    # Transformer inseguro: combinar con VADER
    if conf < 0.60:
        label_v, _ = _vader_label(vader.polarity_scores(limpio))
        if label == label_v:
            return label, conf * 100
        return "Neutral", 60.0

    return label, conf * 100


# ─────────────────────────────────────────────────────────────────
# ANÁLISIS COMPLETO DE UN COMENTARIO
# ─────────────────────────────────────────────────────────────────
def analizar(texto: str, spam_pipe, vader, transformer, batch_spam: bool = False) -> dict:
    texto = str(texto or "").strip()
    if not texto:
        return {"spam": 0, "spam_conf": 0.0, "sentimiento": "Neutral", "sent_conf": 50.0, "motivo": ""}

    # Spam
    if batch_spam:
        spam, spam_conf, motivo = 1, 97.0, "repetición"
    else:
        r_spam, r_conf = reglas_duras(texto)
        if r_spam is not None:
            spam, spam_conf, motivo = int(r_spam), r_conf, "regla" if r_spam else ""
        else:
            probas   = spam_pipe.predict_proba([texto])[0]
            clases   = spam_pipe.named_steps["clf"].classes_
            idx_spam = int(np.where(clases == 1)[0][0]) if 1 in clases else 1
            spam     = int(spam_pipe.predict([texto])[0])
            spam_conf = float(probas[idx_spam]) * 100
            motivo   = "ML" if spam else ""

    sentimiento, sent_conf = analizar_sentimiento(texto, vader, transformer)
    return {"spam": spam, "spam_conf": spam_conf, "sentimiento": sentimiento,
            "sent_conf": sent_conf, "motivo": motivo}


# ─────────────────────────────────────────────────────────────────
# DESCARGA DE COMENTARIOS
# ─────────────────────────────────────────────────────────────────
def descargar_comentarios(url: str, limite: int) -> list[dict]:
    dl  = YoutubeCommentDownloader()
    raw = islice(dl.get_comments_from_url(url, sort_by=SORT_BY_RECENT), limite)
    out = []
    for c in raw:
        texto = (c.get("text") or c.get("content") or "").strip()
        if texto:
            out.append({"Autor": c.get("author", ""), "Comentario": texto, "Fecha": c.get("time", "")})
    return out


# ─────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────
def main():
    st.title("🎬 IA YouTube Global Auditor")

    try:
        df, texto_col, etiqueta_col = cargar_dataset()
    except Exception as e:
        st.error(str(e)); st.stop()

    with st.spinner("Cargando modelos…"):
        spam_pipe, vader, transformer, metricas = cargar_modelos(df)

    # ── Sidebar ──
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=50)
        st.title("YouTube IA Auditor")
        opcion = st.radio("Menú", ["Auditoría Video Real", "Análisis del Dataset", "Prueba Manual"])
        st.divider()
        if opcion == "Auditoría Video Real":
            num_c = st.slider("Comentarios a extraer", 10, 150, 50)
        st.success(f"Spam — Acc {metricas['accuracy']:.2f} · F1 {metricas['f1']:.2f}")
        with st.expander("Reporte de validación"):
            st.text(metricas["reporte"])

    # ── A) Auditoría real ──
    if opcion == "Auditoría Video Real":
        st.header("🎬 Auditoría en Tiempo Real")
        url = st.text_input("URL del vídeo:", placeholder="https://www.youtube.com/watch?v=...")

        if st.button("🚀 Analizar", type="primary"):
            if not url.strip():
                st.warning("Introduce una URL."); return

            with st.spinner("Descargando y analizando…"):
                try:
                    comentarios = descargar_comentarios(url, num_c)
                    if not comentarios:
                        st.error("Sin comentarios (¿desactivados?)."); return

                    flags_batch = detectar_spam_batch(comentarios)

                    filas = []
                    for i, c in enumerate(comentarios):
                        res = analizar(c["Comentario"], spam_pipe, vader, transformer,
                                       batch_spam=flags_batch[i])
                        filas.append({
                            "Autor":        c["Autor"],
                            "Comentario":   c["Comentario"],
                            "Spam":         "🚨 SÍ" if res["spam"] else "✅ NO",
                            "Motivo":       res["motivo"],
                            "Sentimiento":  res["sentimiento"],
                            "Conf. spam":   f"{res['spam_conf']:.0f}%",
                            "Conf. sent.":  f"{res['sent_conf']:.0f}%",
                        })

                    df_res = pd.DataFrame(filas)

                    # KPIs
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Analizados", len(df_res))
                    c2.metric("Spam", f"{(df_res['Spam']=='🚨 SÍ').mean()*100:.1f}%", delta_color="inverse")
                    c3.metric("Audiencia positiva", f"{(df_res['Sentimiento']=='Positive').mean()*100:.1f}%")

                    st.divider()
                    g1, g2 = st.columns(2)

                    with g1:
                        st.plotly_chart(px.pie(
                            df_res, names="Sentimiento", title="Distribución de sentimiento",
                            hole=0.35, color="Sentimiento",
                            color_discrete_map={"Positive":"#2ecc71","Negative":"#e74c3c","Neutral":"#95a5a6"},
                        ), use_container_width=True)

                    with g2:
                        st.write("**Nube de palabras — comentarios reales**")
                        txt = " ".join(df_res.loc[df_res["Spam"]=="✅ NO","Comentario"].astype(str))
                        if txt.strip():
                            wc = WordCloud(background_color="white", collocations=False,
                                           stopwords=set(STOPWORDS)).generate(txt)
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
                            st.pyplot(fig, clear_figure=True)
                        else:
                            st.info("No hay comentarios reales suficientes.")

                    # Desglose de motivos de spam
                    spam_df = df_res[df_res["Spam"]=="🚨 SÍ"]
                    if not spam_df.empty:
                        motivos = spam_df["Motivo"].value_counts().reset_index()
                        motivos.columns = ["Motivo", "Comentarios"]
                        st.plotly_chart(px.bar(motivos, x="Motivo", y="Comentarios",
                                               title="¿Por qué se marcó como spam?"),
                                        use_container_width=True)

                    st.subheader("📋 Detalle completo")
                    st.dataframe(df_res, use_container_width=True)
                    st.download_button("📥 Descargar CSV",
                                       df_res.to_csv(index=False).encode("utf-8"),
                                       "auditoria.csv", "text/csv")

                except Exception as e:
                    st.error(f"Error: {e}"); st.exception(e)

    # ── B) Dataset ──
    elif opcion == "Análisis del Dataset":
        st.header("📊 Dataset de entrenamiento")
        df_v = df.assign(Etiqueta=df["_spam"].map({0:"Real",1:"Spam"}))

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.pie(df_v, names="Etiqueta", title="Balance Real/Spam", hole=0.4),
                            use_container_width=True)
        with col2:
            spam_df = df_v[df_v["_spam"]==1]
            if len(spam_df) >= 2:
                cv   = CountVectorizer(max_features=12, ngram_range=(1,2))
                cnts = cv.fit_transform(spam_df["_texto"])
                w_df = pd.DataFrame({"Término": cv.get_feature_names_out(),
                                     "Frecuencia": cnts.sum(axis=0).A1}).sort_values("Frecuencia")
                st.plotly_chart(px.bar(w_df, x="Frecuencia", y="Término",
                                       orientation="h", title="Top términos en spam"),
                                use_container_width=True)
        st.caption(f"{len(df_v)} muestras · {(df_v['_spam']==1).sum()} spam · {(df_v['_spam']==0).sum()} reales")

    # ── C) Prueba manual ──
    else:
        st.header("🕵️ Prueba manual")
        texto = st.text_area("Escribe un comentario:")

        if st.button("Analizar"):
            if not texto.strip():
                st.warning("Escribe algo primero."); return

            res = analizar(texto, spam_pipe, vader, transformer)
            c1, c2 = st.columns(2)
            c1.metric("Spam", "🚨 SÍ" if res["spam"] else "✅ NO", f"{res['spam_conf']:.0f}%")
            c2.metric("Sentimiento", res["sentimiento"], f"{res['sent_conf']:.0f}%")

            with st.expander("🔍 Features de spam"):
                vals   = HandcraftedSpamFeatures()._f(texto)
                nombres = ["URLs","Ratio MAYÚS","Exclamaciones","!! múltiple","Emojis",
                           "Palabras en CAPS","Diversidad léxica","Hits léxico spam",
                           "Palabras repetidas","Longitud"]
                for n, v in zip(nombres, vals):
                    st.write(f"**{n}**: {v:.3f}")

            with st.expander("🔍 VADER scores"):
                for k, v in vader.polarity_scores(preprocesar_sent(texto)).items():
                    st.write(f"**{k}**: {v:.3f}")

    st.divider()
    st.caption("v5.1 · Duplicados + Reglas + LR regularizada + VADER + Transformer")


if __name__ == "__main__":
    main()
