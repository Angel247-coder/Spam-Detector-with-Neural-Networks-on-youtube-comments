import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neural_network import MLPClassifier
from transformers import pipeline
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
from itertools import islice
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="IA YouTube Global Auditor", page_icon="🎬", layout="wide")

# CSS para Tarjetas KPI
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; color: #ff4b4b; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CARGA DE MODELOS (CACHÉ) ---
@st.cache_resource
def cargar_inteligencia():
    # Cargar Dataset de entrenamiento
    archivo_data = 'Youtube_Unificado_Procesado.csv'
    if not os.path.exists(archivo_data):
        st.error(f"❌ No se encuentra el archivo '{archivo_data}' en tu repositorio.")
        st.stop()
        
    df_base = pd.read_csv(archivo_data).dropna(subset=['CONTENT', 'CLASS'])
    
    # Entrenar Red Neuronal (Spam)
    vec = TfidfVectorizer(stop_words='english', max_features=1500)
    X = vec.fit_transform(df_base['CONTENT'])
    m_spam = MLPClassifier(hidden_layer_sizes=(12, 6), max_iter=300, random_state=42)
    m_spam.fit(X, df_base['CLASS'])
    
    # Modelo Sentimiento (Transformer Multilingüe - Versión 'Student' más ligera)
    m_sent = pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    
    return vec, m_spam, m_sent, df_base

with st.spinner('Cargando Cerebros de IA...'):
    vec, m_spam, m_sent, df_entrenamiento = cargar_inteligencia()

# --- 3. BARRA LATERAL (NAVEGACIÓN) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=50)
    st.title("YouTube IA Auditor")
    opcion = st.radio("Menú Principal", ["Auditoría Video Real", "Análisis del Dataset", "Prueba Manual (Debug)"])
    st.divider()
    if opcion == "Auditoría Video Real":
        num_c = st.slider("Comentarios a extraer", 10, 150, 50)
        st.caption("Nota: Más comentarios requieren más tiempo de proceso.")

# --- 4. FUNCIONES LÓGICAS ---
def analizar_texto(texto):
    v = vec.transform([texto])
    is_spam = m_spam.predict(v)[0]
    conf = m_spam.predict_proba(v)[0][is_spam] * 100
    sent_res = m_sent(texto)[0]
    return is_spam, conf, sent_res['label'].capitalize()

# --- 5. PANTALLAS ---

# A. AUDITORÍA REAL
if opcion == "Auditoría Video Real":
    st.header("🎬 Auditoría de Video en Tiempo Real")
    url = st.text_input("Pega la URL del video:", placeholder="https://www.youtube.com/watch?v=...")
    
    if st.button("🚀 Iniciar Análisis Profundo", type="primary"):
        if not url:
            st.warning("⚠️ Introduce una URL.")
        else:
            with st.spinner("Conectando con YouTube y analizando sentimientos..."):
                try:
                    downloader = YoutubeCommentDownloader()
                    comments = list(islice(downloader.get_comments_from_url(url, sort_by=SORT_BY_RECENT), num_c))
                    
                    if not comments:
                        st.error("No se pudieron obtener comentarios. ¿Están desactivados?")
                    else:
                        res = []
                        for c in comments:
                            is_s, cnf, snt = analizar_texto(c['text'])
                            res.append({
                                'Autor': c['author'], 'Comentario': c['text'],
                                'Tipo': "🚨 Spam" if is_s == 1 else "✅ Real",
                                'Sentimiento': snt, 'Confianza': f"{cnf:.1f}%"
                            })
                        df_res = pd.DataFrame(res)
                        
                        # KPIs
                        k1, k2, k3 = st.columns(3)
                        spam_p = (df_res['Tipo'] == "🚨 Spam").mean() * 100
                        happy_p = (df_res['Sentimiento'] == "Positive").mean() * 100
                        k1.metric("Analizados", len(df_res))
                        k2.metric("Nivel de Spam", f"{spam_p:.1f}%", delta_color="inverse")
                        k3.metric("Felicidad Audiencia", f"{happy_p:.1f}%")

                        # Visualización
                        st.divider()
                        c_a, c_b = st.columns(2)
                        with c_a:
                            st.plotly_chart(px.pie(df_res, names='Sentimiento', title="Clima Emocional", 
                                             color_discrete_map={'Positive':'#2ecc71','Negative':'#e74c3c','Neutral':'#95a5a6'}))
                        with c_b:
                            st.write("**Temas Reales (Nube de palabras)**")
                            txt_r = " ".join(df_res[df_res['Tipo'] == "✅ Real"]['Comentario'])
                            if txt_r:
                                wc = WordCloud(background_color="white", colormap="Blues").generate(txt_r)
                                fig, ax = plt.subplots(); ax.imshow(wc); ax.axis("off"); st.pyplot(fig)

                        st.subheader("📋 Auditoría Detallada")
                        st.dataframe(df_res, use_container_width=True)
                        st.download_button("📥 Descargar Reporte CSV", df_res.to_csv(index=False).encode('utf-8'), "auditoria.csv")

                except Exception as e:
                    st.error(f"Fallo en la conexión: {e}")

# B. ANÁLISIS DEL DATASET
elif opcion == "Análisis del Dataset":
    st.header("📊 Exploración del Dataset de Entrenamiento")
    st.write("Datos cargados de: `Youtube_Unificado_Procesado.csv`")
    
    col1, col2 = st.columns(2)
    with col1:
        df_entrenamiento['Etiqueta'] = df_entrenamiento['CLASS'].map({0: 'Real', 1: 'Spam'})
        st.plotly_chart(px.pie(df_entrenamiento, names='Etiqueta', title="Balance Spam/Real", hole=0.4))
    
    with col2:
        st.write("**Top Palabras en Spam**")
        cv = CountVectorizer(stop_words='english', max_features=10)
        cnts = cv.fit_transform(df_entrenamiento[df_entrenamiento['CLASS']==1]['CONTENT'])
        w_df = pd.DataFrame({'Palabra': cv.get_feature_names_out(), 'Frecuencia': cnts.sum(axis=0).A1}).sort_values('Frecuencia')
        st.plotly_chart(px.bar(w_df, x='Frecuencia', y='Palabra', orientation='h'))

# C. PRUEBA MANUAL
else:
    st.header("🕵️ Prueba Manual de Comentarios")
    test_txt = st.text_area("Escribe algo para probar la IA:")
    if st.button("Analizar"):
        s, c, sn = analizar_texto(test_txt)
        res_c = st.columns(2)
        res_c[0].metric("Detección", "🚨 SPAM" if s == 1 else "✅ REAL", f"{c:.1f}% confianza")
        res_c[1].metric("Sentimiento", sn)

st.divider()
st.caption("Versión 4.0 Pro | Redes Neuronales + Transformers Multilingües")
