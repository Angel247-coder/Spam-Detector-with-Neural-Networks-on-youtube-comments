"""
IA YouTube Spam Detector — v8.1  "Light Dataset"

Fuentes de datos
════════════════════════════════════════════════════════════════════
SPAM (modelo de clasificación)
  1. Youtube-Spam-Dataset.csv        1 956 filas  CONTENT / CLASS (0/1)
  2. YouTube Comments Dataset …csv  45 005 filas  comment_text / label_spam
  ──────────────────────────────────────────────
  Total combinado: ~47 000 filas  │  spam:1 821  real:45 135
  Modelo: MLP(64→32) + TF-IDF bigramas + 9 features del EDA

SENTIMIENTO (modelo de clasificación)
  3. YouTube Comments Dataset …csv  45 005 filas  comment_text / label_sentiment
  ──────────────────────────────────────────────
  Total: ~45 000 filas  │  Positive / Neutral / Negative
  Modelo: LR(C=1, balanced, saga) + TF-IDF bigramas 30k features

ARQUITECTURA (Sprint 3.1 del documento)
  • Spam:       MLP superficial (2 capas ocultas: 64, 32)
  • Sentimiento: Regresión Logística multiclase (rápida, precisa, 3 clases)
  • Ambos modelos cacheados con @st.cache_resource

RGPD (Art. 25, 5.1.c, 5.1.e, 6.1.f)
  Seudonimización SHA-256 inmediata. Sin persistencia. CSV anonimizado.

YouTube ToS
  Auditoría en tiempo real vía YouTube Data API v3 oficial.
"""

import hashlib
import os
import re
from difflib import SequenceMatcher

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from googleapiclient.discovery import build          # pip install google-api-python-client
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils.class_weight import compute_sample_weight
from wordcloud import STOPWORDS, WordCloud


# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="YouTube Spam Detector", page_icon="🎬", layout="wide")
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 26px; color: #ff4b4b; }
.stMetric { background:#fff; padding:14px; border-radius:10px; border:1px solid #eee; }
.rgpd-box { background:#f0f4ff; border-left:4px solid #3b82f6;
            padding:10px 14px; border-radius:6px; font-size:.83rem; margin-bottom:1rem; }
.warn-box  { background:#fff7ed; border-left:4px solid #f59e0b;
             padding:10px 14px; border-radius:6px; font-size:.83rem; margin-bottom:1rem; }
.good-box  { background:#f0fdf4; border-left:4px solid #22c55e;
             padding:10px 14px; border-radius:6px; font-size:.83rem; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# REGEX Y LÉXICO
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
# FEATURES MANUALES PARA SPAM (EDA §3.1)
# ─────────────────────────────────────────────────────────────────
class SpamFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self

    def transform(self, X):
        return np.array([self._f(t) for t in X], dtype=float)

    def _f(self, text: str) -> list:
        t  = str(text)
        tl = t.lower()
        ws = tl.split()
        nc = max(len(t), 1)
        nw = max(len(ws), 1)
        return [
            len(URL_RE.findall(t)),                          # contiene_url (r=0.33)
            sum(c.isupper() for c in t) / nc,               # ratio_mayus  (r=0.05)
            t.count("!"),
            len(EXCL_RE.findall(t)),
            len(CAPS_WORD_RE.findall(t)),
            len(set(ws)) / nw,                              # diversidad léxica
            sum(1 for w in SPAM_LEXICON if w in tl),        # palabras_spam (r=0.69)
            len(REPEAT_RE.findall(tl)),                     # repetición de palabras
            np.log1p(float(len(t))),                        # longitud_chars log (r=0.34)
        ]


# ─────────────────────────────────────────────────────────────────
# RGPD — SEUDONIMIZACIÓN (Art. 25)
# ─────────────────────────────────────────────────────────────────
def seudonimizar(nombre: str) -> str:
    """SHA-256 del nombre real → 'Usr-XXXXXXXX'. Irreversible sin la clave."""
    d = hashlib.sha256(str(nombre).encode("utf-8")).hexdigest()[:8].upper()
    return f"Usr-{d}"


# ─────────────────────────────────────────────────────────────────
# CARGA Y COMBINACIÓN DE LOS DATASETS
# ─────────────────────────────────────────────────────────────────
def _resolver_ruta(*candidatos: str) -> str | None:
    """Devuelve la primera ruta existente de una lista de candidatos.
    Permite que la app encuentre los CSV tanto con el nombre original
    (espacios y paréntesis) como con el nombre saneado que generan
    algunos entornos al subir el archivo (guiones bajos)."""
    for c in candidatos:
        if os.path.exists(c):
            return c
    return None


RUTAS = {
    "spam_clasico": _resolver_ruta(
        "Youtube-Spam-Dataset.csv",
        "data/Youtube-Spam-Dataset.csv",
    ),
    "spam_45k": _resolver_ruta(
        "YouTube Comments Dataset with Sentiment Toxicity and Spam Labels (45K Rows).csv",
        "YouTube_Comments_Dataset_with_Sentiment_Toxicity_and_Spam_Labels__45K_Rows_.csv",
        "data/YouTube Comments Dataset with Sentiment Toxicity and Spam Labels (45K Rows).csv",
        "data/YouTube_Comments_Dataset_with_Sentiment_Toxicity_and_Spam_Labels__45K_Rows_.csv",
    ),
    "export_anterior": _resolver_ruta(
        "2026-05-13T17-45_export.csv",
    ),
}

@st.cache_data(show_spinner=False)
def cargar_datos_spam(ratio_real_spam: int = 1) -> pd.DataFrame:
    """
    Carga y combina los datasets de spam aplicando undersampling estratificado
    sobre la clase mayoritaria (real) para equilibrar el entrenamiento.

    ratio_real_spam: cuántos comentarios reales por cada spam.
        1  →  50 % spam / 50 % real  (F1-spam ≈ 0.85, Prec ≈ 0.81)
        2  →  33 % spam / 67 % real  (F1-spam ≈ 0.82, Prec ≈ 0.77)
        3  →  25 % spam / 75 % real  (F1-spam ≈ 0.75, Prec ≈ 0.67)
    """
    partes = []

    # Fuente 1: dataset clásico (1 956 filas, ~50 % spam)
    if RUTAS["spam_clasico"]:
        df = pd.read_csv(RUTAS["spam_clasico"])[["CONTENT", "CLASS"]].dropna()
        df.columns = ["text", "spam"]
        df["spam"] = df["spam"].astype(int)
        partes.append(df)

    # Fuente 2: dataset 45k moderno (45 005 filas, ~1.8 % spam)
    if RUTAS["spam_45k"]:
        df = pd.read_csv(RUTAS["spam_45k"])[["comment_text", "label_spam"]].dropna()
        df.columns = ["text", "spam"]
        df["spam"] = (df["spam"].str.strip().str.lower() == "spam").astype(int)
        partes.append(df)

    # Fuente 3: export anterior (si tiene columna Spam)
    if RUTAS["export_anterior"]:
        df = pd.read_csv(RUTAS["export_anterior"])
        if "Comentario" in df.columns and "Spam" in df.columns:
            df = df[["Comentario", "Spam"]].dropna()
            df.columns = ["text", "spam"]
            df["spam"] = df["spam"].str.contains("SÍ|1|spam", case=False, na=False).astype(int)
            partes.append(df)

    if not partes:
        raise FileNotFoundError(
            "No se encontró ningún dataset de spam. "
            "Coloca al menos uno de los CSV en la carpeta de la app."
        )

    combined = pd.concat(partes, ignore_index=True).dropna()
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"] != ""]

    # ── Undersampling estratificado de la clase mayoritaria ──────────────
    # Objetivo: ratio_real_spam comentarios reales por cada spam.
    # Esto elimina el desbalance extremo (1:25) que destruía la precisión.
    spam_df = combined[combined["spam"] == 1]
    real_df  = combined[combined["spam"] == 0]

    n_spam    = len(spam_df)
    n_real_target = min(n_spam * ratio_real_spam, len(real_df))

    real_df_sampled = real_df.sample(n=n_real_target, random_state=42)
    balanced = pd.concat([spam_df, real_df_sampled], ignore_index=True).sample(
        frac=1, random_state=42
    )
    return balanced


@st.cache_data(show_spinner=False)
def cargar_datos_sentimiento() -> pd.DataFrame:
    """
    Carga las etiquetas de sentimiento usando únicamente el dataset de 45K.
    """
    partes = []

    # Fuente única: 45k dataset (sentimientos recientes, multilingüe)
    if RUTAS["spam_45k"]:
        df = pd.read_csv(RUTAS["spam_45k"], usecols=["comment_text", "label_sentiment"])
        df.columns = ["text", "sentiment"]
        df["sentiment"] = df["sentiment"].str.lower().str.strip()
        df = df[df["sentiment"].isin(["positive", "neutral", "negative"])].dropna()
        partes.append(df)

    if not partes:
        # Fallback: sin datos etiquetados no se puede entrenar
        raise FileNotFoundError("No se encontró el dataset de sentimiento (45K Rows).")

    combined = pd.concat(partes, ignore_index=True).dropna()
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"] != ""].sample(frac=1, random_state=42)
    return combined


# ─────────────────────────────────────────────────────────────────
# ENTRENAMIENTO — SPAM (MLP superficial)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def entrenar_spam(df: pd.DataFrame):
    X = df["text"].tolist()
    y = df["spam"].tolist()

    sw = compute_sample_weight("balanced", y)

    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_df=0.95,
                max_features=8000, sublinear_tf=True, strip_accents="unicode",
            )),
            ("hc", SpamFeatures()),
        ])),
        ("scaler", MaxAbsScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", alpha=0.01,
            early_stopping=True, validation_fraction=0.12, n_iter_no_change=20,
            max_iter=500, random_state=42,
        )),
    ])

    X_tr, X_val, y_tr, y_val, sw_tr, _ = train_test_split(
        X, y, sw, test_size=0.2, stratify=y, random_state=42
    )
    pipeline.fit(X_tr, y_tr, clf__sample_weight=sw_tr)
    y_pred = pipeline.predict(X_val)

    metricas = {
        "accuracy":  accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall":    recall_score(y_val, y_pred, zero_division=0),
        "f1":        f1_score(y_val, y_pred, zero_division=0),
        "reporte":   classification_report(y_val, y_pred, target_names=["Real", "Spam"], zero_division=0),
        "cm":        confusion_matrix(y_val, y_pred),
        "y_val":     y_val,
        "y_pred":    y_pred,
        "n_train":   len(X_tr),
        "n_spam":    int(sum(y)),
        "n_real":    int(len(y) - sum(y)),
    }
    return pipeline, metricas


# ─────────────────────────────────────────────────────────────────
# ENTRENAMIENTO — SENTIMIENTO (LR multiclase)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def entrenar_sentimiento(df: pd.DataFrame):
    X = df["text"].tolist()
    y = df["sentiment"].tolist()    # positive / neutral / negative

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), max_features=30_000, sublinear_tf=True,
            min_df=3, max_df=0.95, strip_accents="unicode",
        )),
        ("scaler", MaxAbsScaler()),
        ("clf", LogisticRegression(
            C=1.0, max_iter=1000, solver="saga", class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_val)

    metricas = {
        "accuracy":   accuracy_score(y_val, y_pred),
        "f1_macro":   f1_score(y_val, y_pred, average="macro", zero_division=0),
        "reporte":    classification_report(y_val, y_pred, zero_division=0),
        "cm":         confusion_matrix(y_val, y_pred, labels=["positive", "neutral", "negative"]),
        "n_train":    len(X_tr),
    }
    return pipeline, metricas


# ─────────────────────────────────────────────────────────────────
# DETECCIÓN BOT EN BATCH (repetición por seudónimo)
# ─────────────────────────────────────────────────────────────────
def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def detectar_bots(comentarios: list[dict]) -> dict[int, bool]:
    resultado = {i: False for i in range(len(comentarios))}
    por_autor: dict[str, list[int]] = {}
    for i, c in enumerate(comentarios):
        por_autor.setdefault(c.get("seudónimo", ""), []).append(i)
    for _, idx in por_autor.items():
        if len(idx) < 2:
            continue
        textos = [comentarios[i]["texto"] for i in idx]
        total = similares = 0
        for ia in range(len(textos)):
            for ib in range(ia + 1, len(textos)):
                total += 1
                if _sim(textos[ia], textos[ib]) >= 0.80:
                    similares += 1
        if total and (similares / total) >= 0.60:
            for i in idx:
                resultado[i] = True
    return resultado


# ─────────────────────────────────────────────────────────────────
# REGLAS DURAS DE SPAM
# ─────────────────────────────────────────────────────────────────
def reglas_duras(texto: str):
    t, tl = str(texto), str(texto).lower()
    nc = max(len(t), 1)
    if URL_RE.search(t):                                       return True,  95.0
    if sum(1 for w in SPAM_LEXICON if w in tl) >= 3:           return True,  90.0
    if REPEAT_RE.search(tl):                                   return True,  85.0
    if sum(c.isupper() for c in t) / nc > 0.70 and len(t) > 15:   return True,  80.0
    if t.count("!") >= 5:                                      return True,  78.0
    if len(t.split()) <= 2:                                    return False, 80.0
    return None, None


# ─────────────────────────────────────────────────────────────────
# PREPROCESADO TEXTO PARA SENTIMIENTO
# ─────────────────────────────────────────────────────────────────
def preprocesar(texto: str) -> str:
    t = URL_RE.sub(" ", str(texto))
    t = TIMESTAMP_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = EMOJI_RE.sub(" ", t)
    t = REPEAT_CHR_RE.sub(r"\1\1", t)
    return re.sub(r"\s+", " ", t).strip()


# ─────────────────────────────────────────────────────────────────
# ANÁLISIS COMPLETO DE UN COMENTARIO
# ─────────────────────────────────────────────────────────────────
def analizar(texto: str, spam_pipe, sent_pipe, batch_spam: bool = False) -> dict:
    texto = str(texto or "").strip()
    if not texto:
        return {"spam": 0, "spam_conf": 0.0, "sentimiento": "Neutral", "sent_conf": 50.0, "motivo": ""}

    # — SPAM —
    if batch_spam:
        spam, spam_conf, motivo = 1, 97.0, "bot (repetición)"
    else:
        rd, rd_c = reglas_duras(texto)
        if rd is not None:
            spam, spam_conf, motivo = int(rd), rd_c, "regla" if rd else ""
        else:
            probas   = spam_pipe.predict_proba([texto])[0]
            clases   = spam_pipe.named_steps["clf"].classes_
            idx_spam = int(np.where(clases == 1)[0][0]) if 1 in clases else 1
            spam     = int(spam_pipe.predict([texto])[0])
            spam_conf = float(probas[idx_spam]) * 100
            motivo   = "MLP" if spam else ""

    # — SENTIMIENTO —
    limpio     = preprocesar(texto)
    sent_raw   = sent_pipe.predict([limpio] if limpio else [texto])[0]
    sent_conf  = float(sent_pipe.predict_proba([limpio] if limpio else [texto])[0].max()) * 100
    label_map  = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}
    sentimiento = label_map.get(sent_raw, sent_raw.capitalize())

    return {
        "spam": spam, "spam_conf": spam_conf,
        "sentimiento": sentimiento, "sent_conf": sent_conf,
        "motivo": motivo,
    }


# ─────────────────────────────────────────────────────────────────
# YOUTUBE DATA API v3 — DESCARGA OFICIAL
# ─────────────────────────────────────────────────────────────────
def extraer_video_id(url: str) -> str | None:
    m = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([0-9A-Za-z_-]{11})", url)
    return m.group(1) if m else None

def descargar_comentarios(api_key: str, video_id: str, limite: int) -> list[dict]:
    yt = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
    comentarios: list[dict] = []
    page_token = None
    while len(comentarios) < limite:
        por_pagina = min(100, limite - len(comentarios))
        kw = dict(part="snippet", videoId=video_id, maxResults=por_pagina, order="time", textFormat="plainText")
        if page_token:
            kw["pageToken"] = page_token
        resp = yt.commentThreads().list(**kw).execute()
        for item in resp.get("items", []):
            snip  = item["snippet"]["topLevelComment"]["snippet"]
            texto = snip.get("textDisplay", "").strip()
            if texto:
                comentarios.append({
                    "seudónimo": seudonimizar(snip.get("authorDisplayName", "")),
                    "texto": texto,
                })
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return comentarios[:limite]


# ─────────────────────────────────────────────────────────────────
# CARGA DE CSV DEL USUARIO
# ─────────────────────────────────────────────────────────────────
def leer_csv_usuario(upload) -> tuple[pd.DataFrame | None, str | None]:
    try:
        df = pd.read_csv(upload)
    except Exception as e:
        return None, str(e)
    cols = {c.lower(): c for c in df.columns}
    col  = next((cols[k] for k in ("comentario", "content", "text", "comment") if k in cols), None)
    if col is None:
        return None, f"Columnas no reconocidas: {list(df.columns)}"
    df = df.dropna(subset=[col]).copy()
    df["texto"] = df[col].astype(str).str.strip()
    df = df[df["texto"] != ""]
    a = next((cols[k] for k in ("autor", "author") if k in cols), None)
    df["seudónimo"] = df[a].apply(seudonimizar) if a else [f"Usr-{i:04d}" for i in range(len(df))]
    return df, None


# ─────────────────────────────────────────────────────────────────
# GRÁFICAS AUXILIARES
# ─────────────────────────────────────────────────────────────────
def plot_cm(cm, labels, title):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    return fig

def grafico_sentimiento(df_res: pd.DataFrame):
    return px.pie(
        df_res, names="Sentimiento", title="Distribución de sentimiento",
        hole=0.35, color="Sentimiento",
        color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral":  "#95a5a6"},
    )

def nube_palabras(df_res: pd.DataFrame):
    txt = " ".join(df_res.loc[df_res["Spam"] == "✅ NO", "Comentario"].astype(str))
    if not txt.strip():
        return None
    wc = WordCloud(background_color="white", collocations=False, stopwords=set(STOPWORDS)).generate(txt)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

def analizar_batch(comentarios: list[dict], spam_pipe, sent_pipe) -> list[dict]:
    flags = detectar_bots(comentarios)
    filas = []
    for i, c in enumerate(comentarios):
        res = analizar(c["texto"], spam_pipe, sent_pipe, batch_spam=flags[i])
        filas.append({
            "Seudónimo":   c["seudónimo"],
            "Comentario":  c["texto"],
            "Spam":        "🚨 SÍ" if res["spam"] else "✅ NO",
            "Motivo":      res["motivo"],
            "Sentimiento": res["sentimiento"],
            "Conf. spam":  f"{res['spam_conf']:.0f}%",
            "Conf. sent.": f"{res['sent_conf']:.0f}%",
        })
    return filas

def mostrar_resultados(df_res: pd.DataFrame):
    k1, k2, k3 = st.columns(3)
    k1.metric("Analizados",   len(df_res))
    k2.metric("Spam",         f"{(df_res['Spam']=='🚨 SÍ').mean()*100:.1f}%", delta_color="inverse")
    k3.metric("Positivos",    f"{(df_res['Sentimiento']=='Positive').mean()*100:.1f}%")

    st.divider()
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(grafico_sentimiento(df_res), use_container_width=True)
    with g2:
        st.write("**Nube de palabras — comentarios reales**")
        fig_wc = nube_palabras(df_res)
        if fig_wc:
            st.pyplot(fig_wc, clear_figure=True)
        else:
            st.info("No hay suficientes comentarios reales.")

    spam_df = df_res[df_res["Spam"] == "🚨 SÍ"]
    if not spam_df.empty:
        m = spam_df["Motivo"].value_counts().reset_index()
        m.columns = ["Motivo", "N"]
        st.plotly_chart(px.bar(m, x="Motivo", y="N", title="Motivo de clasificación como spam"), use_container_width=True)

    st.subheader("📋 Tabla detallada")
    st.info("🔒 'Seudónimo' = SHA-256. Ningún nombre real en esta tabla ni en el CSV.")
    st.dataframe(df_res, use_container_width=True)
    st.download_button("📥 Descargar CSV anonimizado", df_res.to_csv(index=False).encode("utf-8"), "auditoria_anonimizada.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────
# AVISO DE PRIVACIDAD
# ─────────────────────────────────────────────────────────────────
AVISO_RGPD = """
**Aviso RGPD**
- **Seudonimización inmediata** (Art. 25): SHA-256 → `Usr-XXXXXXXX`
- **Minimización** (Art. 5.1.c): sólo texto y seudónimo
- **Sin persistencia** (Art. 5.1.e): sólo en sesión activa
- **Base jurídica**: interés legítimo para moderación (Art. 6.1.f)
- El CSV exportado no contiene nombres reales
"""

# ─────────────────────────────────────────────────────────────────
# APP PRINCIPAL
# ─────────────────────────────────────────────────────────────────
def main():
    st.title("🎬 YouTube Spam & Sentiment Detector")

    # Cargar datos
    # ratio_sel se define en la sidebar más abajo; necesitamos un default aquí
    # para el primer render. Streamlit re-ejecuta el script completo al cambiar
    # widgets, así que el valor correcto llega en la segunda pasada.
    ratio_sel = st.session_state.get("_ratio_sel", 1)

    with st.spinner("Cargando datasets…"):
        try:
            df_spam = cargar_datos_spam(ratio_real_spam=ratio_sel)
            df_sent = cargar_datos_sentimiento()
        except FileNotFoundError as e:
            st.error(str(e))
            st.info(
                "Coloca los CSV en la misma carpeta que app.py:\n"
                "- `Youtube-Spam-Dataset.csv`\n"
                "- `YouTube Comments Dataset with Sentiment Toxicity and Spam Labels (45K Rows).csv`"
            )
            st.stop()

    # Entrenar modelos
    with st.spinner("Entrenando modelos (primera carga súper rápida)…"):
        spam_pipe, m_spam = entrenar_spam(df_spam)
        sent_pipe, m_sent = entrenar_sentimiento(df_sent)

    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=50)
        st.title("Menú")
        opcion = st.radio("", [
            "🔎 Análisis manual",
            "📂 Análisis por fichero",
            "🎬 Auditoría en tiempo real",
            "📊 Rendimiento de los modelos",
            "📈 Datasets de entrenamiento",
        ])
        st.divider()

        # ── Control de balance del dataset ──────────────────────
        st.subheader("⚖️ Balance del dataset")
        ratio_sel = st.radio(
            "Real : Spam",
            options=[1, 2, 3],
            format_func=lambda r: {
                1: "1:1 — Balanceado (recomendado)",
                2: "2:1 — Moderado",
                3: "3:1 — Conservador",
            }[r],
            index=0,
            help="Ratio de comentarios reales por cada spam en el entrenamiento. "
                 "1:1 maximiza Precisión y F1. Valores mayores aumentan Recall pero bajan Precisión.",
        )
        # Guardar en session_state y limpiar caché si cambia
        if st.session_state.get("_ratio_sel") != ratio_sel:
            st.session_state["_ratio_sel"] = ratio_sel
            cargar_datos_spam.clear()
            entrenar_spam.clear()
        st.caption(
            f"ℹ️ Se usarán **{1821}** spam + **{min(1821*ratio_sel,45135):,}** reales "
            f"→ {1821/(1821+min(1821*ratio_sel,45135))*100:.0f}% spam"
        )
        st.divider()

        # API Key — solo cuando se necesita
        api_key = num_api = ""
        if opcion == "🎬 Auditoría en tiempo real":
            st.subheader("🔑 YouTube Data API v3")
            api_key = st.text_input("API Key", type="password", help="Google Cloud Console → APIs → YouTube Data API v3")
            num_api = st.slider("Comentarios", 10, 200, 50)
            st.caption("🔒 La key sólo existe en esta sesión.")
            st.divider()

        # Métricas rápidas
        st.markdown("**Spam model (MLP)**")
        st.metric("Accuracy",  f"{m_spam['accuracy']:.3f}")
        st.metric("F1 (spam)", f"{m_spam['f1']:.3f}")
        st.divider()
        st.markdown("**Sentiment model (LR)**")
        st.metric("Accuracy",    f"{m_sent['accuracy']:.3f}")
        st.metric("F1 macro",    f"{m_sent['f1_macro']:.3f}")
        st.divider()
        with st.expander("📋 Aviso RGPD"):
            st.markdown(AVISO_RGPD)

    # ════════════════════════════════════════════════════════════
    # A) ANÁLISIS MANUAL
    # ════════════════════════════════════════════════════════════
    if opcion == "🔎 Análisis manual":
        st.header("🔎 Análisis manual")
        st.info("Sin datos personales de terceros. Analiza el texto que escribas aquí.")
        texto = st.text_area("Comentario a analizar:", height=110, placeholder="Escribe cualquier comentario…")

        if st.button("Analizar", type="primary"):
            if not texto.strip():
                st.warning("Escribe un comentario primero."); return

            res = analizar(texto, spam_pipe, sent_pipe)
            c1, c2, c3 = st.columns(3)
            c1.metric("Spam",        "🚨 SÍ" if res["spam"] else "✅ NO", f"{res['spam_conf']:.0f}% confianza")
            c2.metric("Sentimiento", res["sentimiento"], f"{res['sent_conf']:.0f}% confianza")
            c3.metric("Detectado por", res["motivo"] or "—")

            with st.expander("🔍 Features del EDA"):
                vals = SpamFeatures()._f(texto)
                nms  = ["Contiene URL","Ratio mayús","Exclamaciones","!! múltiple",
                        "Palabras CAPS","Diversidad léxica","Hits léxico spam",
                        "Palabras repetidas","Longitud log"]
                for n, v in zip(nms, vals):
                    st.write(f"**{n}**: {v:.3f}")

    # ════════════════════════════════════════════════════════════
    # B) ANÁLISIS POR FICHERO
    # ════════════════════════════════════════════════════════════
    elif opcion == "📂 Análisis por fichero":
        st.header("📂 Análisis por fichero CSV")
        st.markdown('<div class="rgpd-box">🔒 <b>RGPD:</b> Autores seudonimizados (SHA-256) antes de cualquier procesado. CSV exportado sin nombres reales.</div>', unsafe_allow_html=True)
        st.markdown('<div class="warn-box">⚠️ <b>ToS YouTube:</b> Analiza CSV con comentarios ya descargados. Sin conexión a APIs externas.</div>', unsafe_allow_html=True)
        upload = st.file_uploader("CSV con columna **Comentario**, **content**, **text** o **comment**", type=["csv"])
        if upload and st.button("🚀 Analizar fichero", type="primary"):
            df_up, err = leer_csv_usuario(upload)
            if err:
                st.error(err); return
            if df_up is None or df_up.empty:
                st.error("Fichero vacío o sin columna reconocida."); return

            with st.spinner(f"Analizando {len(df_up)} comentarios…"):
                comentarios = df_up[["seudónimo", "texto"]].to_dict("records")
                filas = analizar_batch(comentarios, spam_pipe, sent_pipe)

            mostrar_resultados(pd.DataFrame(filas))

    # ════════════════════════════════════════════════════════════
    # C) AUDITORÍA EN TIEMPO REAL
    # ════════════════════════════════════════════════════════════
    elif opcion == "🎬 Auditoría en tiempo real":
        st.header("🎬 Auditoría en tiempo real")
        st.markdown('<div class="rgpd-box">🔒 <b>RGPD:</b> Autores seudonimizados (SHA-256). Sin persistencia. CSV anonimizado.</div>', unsafe_allow_html=True)
        url = st.text_input("URL del vídeo:", placeholder="https://www.youtube.com/watch?v=...")

        if st.button("🚀 Analizar vídeo", type="primary"):
            if not api_key.strip():
                st.warning("Introduce tu API Key en el panel lateral."); return
            if not url.strip():
                st.warning("Introduce la URL del vídeo."); return
            video_id = extraer_video_id(url)
            if not video_id:
                st.error("No se pudo extraer el ID del vídeo."); return

            with st.spinner("Descargando vía YouTube Data API v3…"):
                try:
                    comentarios = descargar_comentarios(api_key, video_id, num_api)
                except Exception as e:
                    st.error(f"Error de la API: {e}")
                    st.info("Comprueba la API Key, que YouTube Data API v3 esté habilitada y que no hayas superado la cuota (10k unidades/día).")
                    return

            if not comentarios:
                st.error("Sin comentarios (¿están desactivados?)."); return

            with st.spinner(f"Analizando {len(comentarios)} comentarios…"):
                filas = analizar_batch(comentarios, spam_pipe, sent_pipe)

            mostrar_resultados(pd.DataFrame(filas))

    # ════════════════════════════════════════════════════════════
    # D) RENDIMIENTO DE LOS MODELOS
    # ════════════════════════════════════════════════════════════
    elif opcion == "📊 Rendimiento de los modelos":
        st.header("📊 Rendimiento de los modelos")

        tab_spam, tab_sent = st.tabs(["🛡️ Spam (MLP)", "💬 Sentimiento (LR)"])

        with tab_spam:
            st.markdown(f"Entrenado con **{m_spam['n_train']:,}** muestras ({m_spam['n_spam']:,} spam · {m_spam['n_real']:,} reales). Holdout estratificado 20%.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy",  f"{m_spam['accuracy']:.3f}")
            c2.metric("Precision", f"{m_spam['precision']:.3f}")
            c3.metric("Recall",    f"{m_spam['recall']:.3f}")
            c4.metric("F1 spam",   f"{m_spam['f1']:.3f}")
            st.divider()
            col_cm, col_rep = st.columns([1, 1.5])
            with col_cm:
                st.pyplot(plot_cm(m_spam["cm"], ["Real", "Spam"], "Confusión spam (20% holdout)"))
            with col_rep:
                st.code(m_spam["reporte"])
            st.divider()
            st.markdown("""
| Parámetro | Valor |
|---|---|
| Tipo | Red Neuronal Superficial (MLP) |
| Capas | 64 → 32 neuronas · ReLU |
| Regularización L2 | alpha = 0.01 |
| Early stopping | Sí (12% val interna) |
| Desbalance | Undersampling estratificado (ratio configurable) |
| Vectorización | TF-IDF bigramas (8k features) |
| Features EDA | 9 (§3.1 del documento) |
| Split | 80 / 20 estratificado |
| Datos spam | 45K dataset (2025) + dataset clásico |
            """)

        with tab_sent:
            st.markdown(f"Entrenado con **{m_sent['n_train']:,}** muestras (basado en el 45K dataset). Holdout 15%.")
            c1, c2 = st.columns(2)
            c1.metric("Accuracy",  f"{m_sent['accuracy']:.3f}")
            c2.metric("F1 macro",  f"{m_sent['f1_macro']:.3f}")
            st.divider()
            col_cm, col_rep = st.columns([1, 1.5])
            with col_cm:
                st.pyplot(plot_cm(m_sent["cm"], ["positive", "neutral", "negative"], "Confusión sentimiento (15% holdout)"))
            with col_rep:
                st.code(m_sent["reporte"])
            st.markdown('<div class="good-box">ℹ️ El modelo utiliza el dataset multilingüe de 45K para entrenar de forma eficiente manteniendo un bajo consumo de memoria.</div>', unsafe_allow_html=True)
            st.divider()
            st.markdown("""
| Parámetro | Valor |
|---|---|
| Tipo | Regresión Logística multiclase |
| Solver | SAGA (rápido) |
| Regularización | C = 1.0 · class_weight = balanced |
| Vectorización | TF-IDF bigramas (30k features) |
| Clases | Positive · Neutral · Negative |
| Split | 85 / 15 estratificado |
| Datos | 45k dataset |
            """)

    # ════════════════════════════════════════════════════════════
    # E) DATASETS DE ENTRENAMIENTO
    # ════════════════════════════════════════════════════════════
    elif opcion == "📈 Datasets de entrenamiento":
        st.header("📈 Datasets de entrenamiento")

        tab_s, tab_sent2 = st.tabs(["🛡️ Spam", "💬 Sentimiento"])

        with tab_s:
            df_s = df_spam.copy()
            df_s["Etiqueta"] = df_s["spam"].map({0: "Real", 1: "Spam"})
            n_s, n_r = (df_s["spam"] == 1).sum(), (df_s["spam"] == 0).sum()
            k1, k2, k3 = st.columns(3)
            k1.metric("Total", f"{len(df_s):,}")
            k2.metric("Spam",  f"{n_s:,} ({n_s/len(df_s)*100:.1f}%)")
            k3.metric("Real",  f"{n_r:,} ({n_r/len(df_s)*100:.1f}%)")

            ratio_labels = {1: "1:1 — Balanceado", 2: "2:1 — Moderado", 3: "3:1 — Conservador"}
            st.markdown(
                f'<div class="good-box">⚖️ <b>Undersampling activo:</b> Ratio {ratio_labels.get(ratio_sel, ratio_sel)} '
                f'· {n_s:,} spam + {n_r:,} reales seleccionados de 45,135 disponibles. '
                f'Sin equilibrar, la precisión era solo <b>0.25</b>; con ratio 1:1 sube a <b>~0.81</b>.</div>',
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.pie(df_s, names="Etiqueta", title="Balance Real / Spam (tras undersampling)", hole=0.4, color="Etiqueta", color_discrete_map={"Real":"#2ecc71","Spam":"#e74c3c"}), use_container_width=True)
            with c2:
                df_s["longitud"] = df_s["text"].str.len()
                st.plotly_chart(px.box(df_s, x="Etiqueta", y="longitud", title="Longitud de comentario por clase", color="Etiqueta", color_discrete_map={"Real":"#2ecc71","Spam":"#e74c3c"}), use_container_width=True)

        with tab_sent2:
            df_sv = df_sent.copy()
            df_sv["Etiqueta"] = df_sv["sentiment"].str.capitalize()
            dist = df_sv["Etiqueta"].value_counts()
            k1, k2, k3 = st.columns(3)
            k1.metric("Total",    f"{len(df_sv):,}")
            k2.metric("Positivos",f"{dist.get('Positive',0):,}")
            k3.metric("Negativos",f"{dist.get('Negative',0):,}")
            st.plotly_chart(px.pie(df_sv, names="Etiqueta", title="Distribución de sentimiento en training (45K Dataset)", hole=0.4, color="Etiqueta", color_discrete_map={"Positive":"#2ecc71", "Negative":"#e74c3c", "Neutral":"#95a5a6"}), use_container_width=True)

    st.divider()
    st.caption("v8.2 (Balanced) · Undersampling 1:1 · MLP spam · LR sentimiento · YouTube Data API v3 · RGPD Art. 25")

if __name__ == "__main__":
    main()
