import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neural_network import MLPClassifier
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN Y ESTILO ---
st.set_page_config(page_title="IA YouTube Global Analyzer", page_icon="🌎", layout="wide")

# CSS para que las tarjetas de métricas resalten
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 30px; color: #1f77b4; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #e6e9ef; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CARGA DE MODELOS (CACHÉ) ---
@st.cache_resource
def cargar_inteligencia():
    # A. Cargar Datos
    df = pd.read_csv('Youtube_Unificado_Procesado.csv')
    df = df.dropna(subset=['CONTENT', 'CLASS'])
    df['Etiqueta'] = df['CLASS'].map({0: 'Real ✅', 1: 'Spam 🚨'})

    # B. IA de Sentimientos Multilingüe (Transformer)
    modelo_sent = pipeline(
        "sentiment-analysis", 
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )

    # C. IA de Spam (Red Neuronal)
    vectorizador = TfidfVectorizer(stop_words='english', max_features=2000)
    X = vectorizador.fit_transform(df['CONTENT'])
    y = df['CLASS']
    modelo_spam = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
    modelo_spam.fit(X, y)

    return vectorizador, modelo_spam, df, modelo_sent

with st.spinner('Sincronizando neuronas...'):
    vectorizador, modelo_spam, df, sentiment_pipe = cargar_inteligencia()

# --- 3. LÓGICA DE CÁLCULO ---
def analizar_comentario(texto):
    vec = vectorizador.transform([texto])
    es_spam = modelo_spam.predict(vec)[0]
    probs = modelo_spam.predict_proba(vec)[0]
    res_sent = sentiment_pipe(texto)[0]
    return es_spam, probs[es_spam] * 100, res_sent['label'].capitalize()

# --- 4. INTERFAZ ---
st.title("🌎 IA YouTube Sentiment & Spam Analyzer")
st.write("Panel de control inteligente para la moderación y análisis de audiencia.")
st.divider()

tab1, tab2, tab3 = st.tabs(["🕵️‍♂️ Analizador Individual", "📊 Dashboard de Métricas", "☁️ Nubes de Aspectos"])

# === PESTAÑA 1: ANALIZADOR ===
with tab1:
    c1, c2 = st.columns([2, 1])
    with c1:
        user_input = st.text_area("Comentario a examinar:", placeholder="Ej: Awesome content! / Suscríbete a mi canal...")
        if st.button("Ejecutar IA", type="primary"):
            if user_input.strip():
                spam, conf, sent = analizar_comentario(user_input)
                with c2:
                    st.subheader("Resultado")
                    if spam == 1: st.error(f"🚨 SPAM ({conf:.1f}%)")
                    else: st.success(f"✅ REAL ({conf:.1f}%)")
                    
                    icon = "🌟" if sent == "Positive" else "💢" if sent == "Negative" else "😶"
                    st.info(f"{icon} Sentimiento: **{sent}**")
                    st.progress(conf/100)

# === PESTAÑA 2: DASHBOARD (MÉTRICAS KPI) ===
with tab2:
    # --- FILA DE MÉTRICAS (KPIs) ---
    st.subheader("📈 Indicadores Clave de Desempeño")
    
    # Cálculos rápidos
    total_comentarios = len(df)
    total_spam = df[df['CLASS'] == 1].shape[0]
    pct_spam = (total_spam / total_comentarios) * 100
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Comentarios Totales", f"{total_comentarios:,}")
    m2.metric("Tasa de Spam", f"{pct_spam:.1f}%", delta=f"{total_spam} detectados", delta_color="inverse")
    m3.metric("Idioma Predominante", "En/Es", delta="Multilingüe activo")
    
    st.divider()
    
    # Gráficos
    col_a, col_b = st.columns(2)
    with col_a:
        fig_pie = px.pie(df, names='Etiqueta', title="Balance de Spam", hole=0.5, color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_b:
        # Mostramos palabras más comunes en el dataset general
        st.write("**Top 10 términos en el canal**")
        cv = CountVectorizer(stop_words='english', max_features=10)
        counts = cv.fit_transform(df['CONTENT'])
        w_df = pd.DataFrame({'Palabra': cv.get_feature_names_out(), 'Frecuencia': counts.sum(axis=0).A1}).sort_values('Frecuencia')
        st.plotly_chart(px.bar(w_df, x='Frecuencia', y='Palabra', orientation='h'), use_container_width=True)

# === PESTAÑA 3: NUBES DE PALABRAS ===
with tab3:
    st.subheader("Análisis Visual de Aspectos")
    c_spam, c_real = st.columns(2)
    
    with c_spam:
        st.write("🚨 **Tendencias en Spam**")
        wc_s = WordCloud(background_color="white", colormap="Reds").generate(" ".join(df[df['CLASS'] == 1]['CONTENT']))
        fig_s, ax_s = plt.subplots(); ax_s.imshow(wc_s); ax_s.axis("off")
        st.pyplot(fig_s)

    with c_real:
        st.write("✅ **Temas de Interés Real**")
        wc_r = WordCloud(background_color="white", colormap="Blues").generate(" ".join(df[df['CLASS'] == 0]['CONTENT']))
        fig_r, ax_r = plt.subplots(); ax_r.imshow(wc_r); ax_r.axis("off")
        st.pyplot(fig_r)

# --- FOOTER ---
st.divider()
st.caption("🚀 Neural Network (Scikit-Learn) + Transformer (Hugging Face) | v3.0")
