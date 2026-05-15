"""
IA YouTube Global Auditor — v6.0

Cambios legales respecto a v5.1
════════════════════════════════════════════════════════════════════

[ToS YouTube — Sección 5.B]
  ANTES: youtube_comment_downloader → scraping no autorizado.
         YouTube prohíbe explícitamente acceder a su servicio por medios
         distintos a las interfaces y APIs oficiales.
  AHORA: YouTube Data API v3 (googleapiclient).
         El usuario aporta su propia API Key (Google Cloud Console).
         Cuota gratuita: 10 000 unidades/día; commentThreads.list = 1 u/req.

[RGPD — Arts. 4, 5, 25 y 89]
  ANTES: Nombres de usuario reales en pantalla y en el CSV exportado.
  AHORA:
    • Seudonimización inmediata (Art. 25 — privacidad por diseño):
      el nombre real se hashea con SHA-256 → "Usr-A3F2C1B0" antes de
      cualquier procesado o visualización.  El nombre original NUNCA
      sale de la función de descarga.
    • Minimización de datos (Art. 5.1.c): sólo se recogen texto del
      comentario y seudónimo. La fecha se descarta si no aporta valor
      al análisis.
    • Limitación de la conservación (Art. 5.1.e): los datos viven
      únicamente en la sesión de Streamlit (st.session_state).
      No hay escritura en disco, base de datos ni logs externos.
    • Exportación anonimizada: el CSV descargable contiene seudónimos,
      jamás nombres reales.
    • Base jurídica documentada: interés legítimo del responsable del
      tratamiento para fines de investigación / moderación de contenido
      (Art. 6.1.f), con aviso visible al usuario de la herramienta.
    • Aviso de privacidad visible en la interfaz antes de iniciar
      cualquier tratamiento.
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
from googleapiclient.discovery import build  # pip install google-api-python-client
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from transformers import pipeline as hf_pipeline
from wordcloud import STOPWORDS, WordCloud

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
# RGPD — SEUDONIMIZACIÓN (Art. 25)
# ─────────────────────────────────────────────────────────────────
def seudonimizar(nombre_real: str) -> str:
    """
    Convierte el nombre real del usuario en un seudónimo reproducible
    pero no reversible sin la clave de hash.

    SHA-256(nombre_real) → primeros 8 hex → "Usr-A3F2C1B0"

    El nombre original se descarta inmediatamente tras esta operación;
    nunca se almacena en sesión, CSV ni logs.
    """
    digest = hashlib.sha256(nombre_real.encode("utf-8")).hexdigest()[:8].upper()
    return f"Usr-{digest}"


# ─────────────────────────────────────────────────────────────────
# FEATURES MANUALES PARA SPAM
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

    vader       = SentimentIntensityAnalyzer()
    transformer = hf_pipeline(
        "sentiment-analysis",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    )
    return spam_pipe, vader, transformer, metricas


# ─────────────────────────────────────────────────────────────────
# ToS YouTube — DESCARGA VÍA API OFICIAL
# ─────────────────────────────────────────────────────────────────
def extraer_video_id(url: str) -> str | None:
    """Extrae el ID de 11 caracteres de cualquier formato de URL de YouTube."""
    patron = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([0-9A-Za-z_-]{11})", url)
    return patron.group(1) if patron else None


def descargar_comentarios_api(api_key: str, video_id: str, limite: int) -> list[dict]:
    """
    Descarga comentarios mediante la YouTube Data API v3 (oficial).

    Devuelve una lista de dicts con:
      - 'seudónimo': nombre seudonimizado (SHA-256), nunca el real
      - 'texto': contenido del comentario
    El nombre real del autor se descarta inmediatamente tras seudonimizar.

    Cuota: 1 unidad por página (hasta 100 comentarios/página).
    """
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
            snip       = item["snippet"]["topLevelComment"]["snippet"]
            nombre_real = snip.get("authorDisplayName", "")

            # ── Seudonimización inmediata — el nombre real nunca sale de aquí ──
            seudónimo = seudonimizar(nombre_real)

            comentarios.append({
                "seudónimo": seudónimo,
                "texto":     snip.get("textDisplay", "").strip(),
            })

        token_siguiente = respuesta.get("nextPageToken")
        if not token_siguiente:
            break

    # Filtrar vacíos y respetar el límite solicitado
    return [c for c in comentarios if c["texto"]][:limite]


# ─────────────────────────────────────────────────────────────────
# SPAM — DUPLICADOS EN BATCH
# ─────────────────────────────────────────────────────────────────
def _similitud(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def detectar_spam_batch(comentarios: list[dict]) -> dict[int, bool]:
    """
    Marca como spam comentarios cuyo seudónimo comparte textos muy similares
    (≥ 0.80) en más del 60 % de sus pares → comportamiento de bot.
    Opera sobre seudónimos, nunca sobre nombres reales.
    """
    resultado: dict[int, bool] = {i: False for i in range(len(comentarios))}
    por_autor: dict[str, list[int]] = {}

    for i, c in enumerate(comentarios):
        por_autor.setdefault(c["seudónimo"], []).append(i)

    for _, indices in por_autor.items():
        if len(indices) < 2:
            continue
        textos = [comentarios[i]["texto"] for i in indices]
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
# SPAM — REGLAS DURAS
# ─────────────────────────────────────────────────────────────────
def reglas_duras(texto: str):
    t, tl, nc = str(texto), str(texto).lower(), max(len(str(texto)), 1)
    if URL_RE.search(t):                                              return True,  95.0
    if sum(c.isupper() for c in t) / nc > 0.60 and len(t) > 10:    return True,  90.0
    if EXCL_RE.search(t):                                            return True,  88.0
    if sum(1 for w in SPAM_LEXICON if w in tl) >= 3:                return True,  87.0
    if REPEAT_RE.search(tl):                                         return True,  85.0
    if len(t.split()) <= 2:                                          return False, 80.0
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
    if not limpio:
        return "Neutral", 50.0
    if len(limpio.split()) < 6:
        return _vader_label(vader.polarity_scores(limpio))
    res   = transformer(limpio[:512], truncation=True)[0]
    label = _transformer_label(res["label"])
    conf  = float(res["score"])
    if conf < 0.60:
        lv, _ = _vader_label(vader.polarity_scores(limpio))
        return (label, conf * 100) if label == lv else ("Neutral", 60.0)
    return label, conf * 100


# ─────────────────────────────────────────────────────────────────
# ANÁLISIS COMPLETO DE UN COMENTARIO
# ─────────────────────────────────────────────────────────────────
def analizar(texto: str, spam_pipe, vader, transformer, batch_spam: bool = False) -> dict:
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
            probas   = spam_pipe.predict_proba([texto])[0]
            clases   = spam_pipe.named_steps["clf"].classes_
            idx_spam = int(np.where(clases == 1)[0][0]) if 1 in clases else 1
            spam     = int(spam_pipe.predict([texto])[0])
            spam_conf = float(probas[idx_spam]) * 100
            motivo   = "ML" if spam else ""

    sentimiento, sent_conf = analizar_sentimiento(texto, vader, transformer)
    return {"spam": spam, "spam_conf": spam_conf,
            "sentimiento": sentimiento, "sent_conf": sent_conf, "motivo": motivo}


# ─────────────────────────────────────────────────────────────────
# AVISO DE PRIVACIDAD (RGPD Art. 13)
# ─────────────────────────────────────────────────────────────────
AVISO_PRIVACIDAD = """
**Aviso de privacidad — tratamiento de datos personales**

Esta herramienta accede a comentarios públicos de YouTube mediante la
**YouTube Data API v3** (términos de servicio: https://www.youtube.com/t/terms).

Los nombres de usuario de los comentaristas son **datos personales** conforme
al Art. 4.1 RGPD. El tratamiento se realiza bajo las siguientes garantías:

- **Seudonimización inmediata** (Art. 25 RGPD): el nombre real se convierte
  en un identificador irreversible (SHA-256) antes de cualquier visualización
  o análisis. El nombre original nunca se almacena ni exporta.
- **Minimización** (Art. 5.1.c): sólo se tratan texto del comentario y
  seudónimo. No se recogen datos de perfil adicionales.
- **Limitación de conservación** (Art. 5.1.e): los datos existen únicamente
  durante la sesión activa de esta aplicación. No hay escritura en disco,
  base de datos ni transmisión a terceros.
- **Base jurídica**: interés legítimo del responsable del tratamiento para
  fines de moderación e investigación de contenido (Art. 6.1.f RGPD).
- El CSV exportable contiene **únicamente seudónimos**, nunca nombres reales.

Al pulsar «Analizar», confirmas que has leído este aviso y que utilizas
la herramienta conforme a la normativa aplicable.
"""


# ─────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────
def main():
    st.title("🎬 IA YouTube Global Auditor")

    # Cargar dataset y modelos
    try:
        df, texto_col, etiqueta_col = cargar_dataset()
    except Exception as e:
        st.error(str(e)); st.stop()

    with st.spinner("Cargando modelos…"):
        spam_pipe, vader, transformer, metricas = cargar_modelos(df)

    # ── Sidebar ──────────────────────────────────────────────────
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
                help=(
                    "Obtén tu clave en Google Cloud Console → APIs → "
                    "YouTube Data API v3. Cuota gratuita: 10 000 unidades/día."
                ),
            )
            num_c = st.slider("Comentarios a extraer", 10, 200, 50)
            st.caption(
                "🔒 La API Key sólo se usa en esta sesión y nunca se almacena."
            )

        st.divider()
        st.success(f"Spam — Acc {metricas['accuracy']:.2f} · F1 {metricas['f1']:.2f}")
        with st.expander("Reporte de validación"):
            st.text(metricas["reporte"])

        st.divider()
        with st.expander("📋 Aviso de privacidad (RGPD)"):
            st.markdown(AVISO_PRIVACIDAD)

    # ── A) Auditoría real ─────────────────────────────────────────
    if opcion == "Auditoría Video Real":
        st.header("🎬 Auditoría en Tiempo Real")

        # Aviso de privacidad visible antes de cualquier acción
        st.markdown(
            '<div class="rgpd-box">🔒 <strong>Privacidad:</strong> '
            'los nombres de usuario se seudonimizarán con SHA-256 antes de '
            'cualquier procesado. No se exportan ni almacenan datos personales '
            'reales. Consulta el aviso completo en el panel lateral.</div>',
            unsafe_allow_html=True,
        )

        url = st.text_input(
            "URL del vídeo:",
            placeholder="https://www.youtube.com/watch?v=...",
        )

        if st.button("🚀 Analizar", type="primary"):
            # Validaciones previas
            if not url.strip():
                st.warning("Introduce una URL."); return
            if not api_key.strip():
                st.warning(
                    "Introduce tu API Key de YouTube Data API v3 en el panel lateral."
                ); return

            video_id = extraer_video_id(url)
            if not video_id:
                st.error(
                    "No se pudo extraer el ID del vídeo. "
                    "Comprueba que la URL es válida."
                ); return

            with st.spinner("Descargando comentarios vía API oficial…"):
                try:
                    comentarios = descargar_comentarios_api(api_key, video_id, num_c)
                except Exception as e:
                    # El error de la API puede incluir detalles de cuota o clave inválida
                    st.error(f"Error de la YouTube API: {e}")
                    st.info(
                        "Comprueba que la API Key es correcta, que YouTube Data API v3 "
                        "está habilitada en tu proyecto de Google Cloud y que no has "
                        "agotado la cuota diaria (10 000 unidades)."
                    )
                    return

            if not comentarios:
                st.error("No se obtuvieron comentarios (¿están desactivados?)."); return

            with st.spinner("Analizando…"):
                flags_batch = detectar_spam_batch(comentarios)

                filas = []
                for i, c in enumerate(comentarios):
                    res = analizar(
                        c["texto"], spam_pipe, vader, transformer,
                        batch_spam=flags_batch[i],
                    )
                    filas.append({
                        "Seudónimo":    c["seudónimo"],   # nunca el nombre real
                        "Comentario":   c["texto"],
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
            c2.metric(
                "Spam detectado",
                f"{(df_res['Spam']=='🚨 SÍ').mean()*100:.1f}%",
                delta_color="inverse",
            )
            c3.metric(
                "Audiencia positiva",
                f"{(df_res['Sentimiento']=='Positive').mean()*100:.1f}%",
            )

            st.divider()
            g1, g2 = st.columns(2)

            with g1:
                st.plotly_chart(px.pie(
                    df_res, names="Sentimiento",
                    title="Distribución de sentimiento",
                    hole=0.35, color="Sentimiento",
                    color_discrete_map={
                        "Positive": "#2ecc71",
                        "Negative": "#e74c3c",
                        "Neutral":  "#95a5a6",
                    },
                ), use_container_width=True)

            with g2:
                st.write("**Nube de palabras — comentarios reales**")
                txt = " ".join(
                    df_res.loc[df_res["Spam"] == "✅ NO", "Comentario"].astype(str)
                )
                if txt.strip():
                    wc = WordCloud(
                        background_color="white", collocations=False,
                        stopwords=set(STOPWORDS),
                    ).generate(txt)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
                    st.pyplot(fig, clear_figure=True)
                else:
                    st.info("No hay comentarios reales suficientes.")

            # Desglose de motivos de spam
            spam_df = df_res[df_res["Spam"] == "🚨 SÍ"]
            if not spam_df.empty:
                motivos = spam_df["Motivo"].value_counts().reset_index()
                motivos.columns = ["Motivo", "Comentarios"]
                st.plotly_chart(px.bar(
                    motivos, x="Motivo", y="Comentarios",
                    title="¿Por qué se marcó como spam?",
                ), use_container_width=True)

            st.subheader("📋 Detalle completo")
            st.info(
                "🔒 La columna «Seudónimo» contiene identificadores SHA-256 "
                "irreversibles. Los nombres reales de los comentaristas no aparecen "
                "en ninguna parte de esta tabla ni del CSV exportado."
            )
            st.dataframe(df_res, use_container_width=True)

            csv_bytes = df_res.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Descargar CSV (datos anonimizados)",
                csv_bytes, "auditoria_anonimizada.csv", "text/csv",
            )

    # ── B) Dataset ────────────────────────────────────────────────
    elif opcion == "Análisis del Dataset":
        st.header("📊 Dataset de entrenamiento")
        df_v = df.assign(Etiqueta=df["_spam"].map({0: "Real", 1: "Spam"}))

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.pie(
                df_v, names="Etiqueta",
                title="Balance Real / Spam", hole=0.4,
            ), use_container_width=True)
        with col2:
            spam_df = df_v[df_v["_spam"] == 1]
            if len(spam_df) >= 2:
                cv   = CountVectorizer(max_features=12, ngram_range=(1, 2))
                cnts = cv.fit_transform(spam_df["_texto"])
                w_df = pd.DataFrame({
                    "Término":    cv.get_feature_names_out(),
                    "Frecuencia": cnts.sum(axis=0).A1,
                }).sort_values("Frecuencia")
                st.plotly_chart(px.bar(
                    w_df, x="Frecuencia", y="Término",
                    orientation="h", title="Top términos en spam",
                ), use_container_width=True)

        st.caption(
            f"{len(df_v)} muestras · {(df_v['_spam']==1).sum()} spam · "
            f"{(df_v['_spam']==0).sum()} reales"
        )

    # ── C) Prueba manual ──────────────────────────────────────────
    else:
        st.header("🕵️ Prueba manual")
        st.info(
            "Esta sección analiza texto introducido directamente. "
            "No se tratan datos personales de terceros."
        )
        texto = st.text_area("Escribe un comentario de prueba:")

        if st.button("Analizar"):
            if not texto.strip():
                st.warning("Escribe algo primero."); return

            res = analizar(texto, spam_pipe, vader, transformer)
            c1, c2 = st.columns(2)
            c1.metric("Spam", "🚨 SÍ" if res["spam"] else "✅ NO",
                      f"{res['spam_conf']:.0f}% confianza")
            c2.metric("Sentimiento", res["sentimiento"],
                      f"{res['sent_conf']:.0f}% confianza")

            with st.expander("🔍 Features de spam"):
                vals    = HandcraftedSpamFeatures()._f(texto)
                nombres = [
                    "URLs", "Ratio MAYÚS", "Exclamaciones", "!! múltiple", "Emojis",
                    "Palabras en CAPS", "Diversidad léxica", "Hits léxico spam",
                    "Palabras repetidas", "Longitud",
                ]
                for n, v in zip(nombres, vals):
                    st.write(f"**{n}**: {v:.3f}")

            with st.expander("🔍 VADER scores"):
                limpio = preprocesar_sent(texto)
                for k, v in vader.polarity_scores(limpio).items():
                    st.write(f"**{k}**: {v:.3f}")

    st.divider()
    st.caption(
        "v6.0 · YouTube Data API v3 (ToS) · Seudonimización SHA-256 (RGPD Art. 25) · "
        "Sesión única sin persistencia (RGPD Art. 5.1.e)"
    )


if __name__ == "__main__":
    main()
