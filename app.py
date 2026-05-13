import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neural_network import MLPClassifier
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN Y ESTILO ---
st.set_page_config(page_title="IA YouTube Analyzer Pro", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. FUNCIONES DE APOYO (IA Y SENTIMIENTO) ---
def obtener_sentimiento(texto):
    # Traducir mentalmente o usar TextBlob (funciona mejor en inglés, pero sirve para términos comunes)
    analisis = TextBlob(texto)
    if analisis.sentiment.polarity > 0.1: return 'Positivo'
    elif analisis.sentiment.polarity < -0.1: return 'Negativo'
    else: return 'Neutral'

@st.cache_resource
def preparar_todo():
    # Cargar datos
    df = pd.read_csv('Youtube_Unificado_Procesado.csv')
    df = df.dropna(subset=['CONTENT', 'CLASS'])
    
    # Análisis de sentimiento previo al dataset
    df['Sentimiento'] = df['CONTENT'].apply(obtener_sentimiento)
    df['Etiqueta'] = df['CLASS'].map({0: 'Real ✅', 1: 'Spam 🚨'})

    # Entrenar Red Neuronal
    vectorizador = TfidfVectorizer(stop_words='english')
    X = vectorizador.fit_transform(df['CONTENT'])
    y = df['CLASS']
    modelo = MLPClassifier(hidden_layer_sizes=(12, 8), max_iter=500, random_state=42)
    modelo.fit(X, y)

    return vectorizador, modelo, df

with st.spinner('Entrenando IA y analizando sentimientos...'):
    vectorizador, modelo, df = preparar_todo()

# --- 3. INTERFAZ PRINCIPAL ---
st.title("🤖 IA YouTube Sentiment & Spam Analyzer")
st.write("Herramienta avanzada de análisis de datos para creadores de contenido.")
st.divider()

# Definir pestañas
tab1, tab2, tab3 = st.tabs(["🕵️‍♂️ Analizador IA", "📊 Dashboard Global", "☁️ Nubes de Aspectos"])

# ==========================================
# PESTAÑA 1: ANALIZADOR CON CONFIANZA
# ==========================================
with tab1:
    col_in, col_out = st.columns([2, 1])
    
    with col_in:
        st.subheader("Analizar Comentario Nuevo")
        texto = st.text_area("Pega el texto aquí:", placeholder="Ej: Great video, thanks for the help!", height=150)
        analizar = st.button("Ejecutar Análisis Completo", type="primary")

    with col_out:
        if analizar and texto.strip() != "":
            # Predicción IA
            vec = vectorizador.transform([texto])
            pred = modelo.predict(vec)[0]
            probs = modelo.predict_proba(vec)[0]
            conf = probs[pred] * 100
            
            # Sentimiento
            sent = obtener_sentimiento(texto)
            
            st.markdown("### Resultados")
            # Mostrar Spam
            if pred == 1: st.error(f"**TIPO:** SPAM (Confianza: {conf:.1f}%)")
            else: st.success(f"**TIPO:** REAL (Confianza: {conf:.1f}%)")
            st.progress(conf/100)
            
            # Mostrar Sentimiento
            st.info(f"**SENTIMIENTO:** {sent}")
        else:
            st.info("Escribe un comentario para ver el análisis de la red neuronal.")

# ==========================================
# PESTAÑA 2: DASHBOARD
# ==========================================
with tab2:
    st.subheader("Resumen de los Datos")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**Distribución de Spam**")
        fig1 = px.pie(df, names='Etiqueta', hole=.4, color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig1, use_container_width=True)
    
    with c2:
        st.markdown("**Sentimientos del Canal**")
        fig2 = px.bar(df['Sentimiento'].value_counts(), color_discrete_sequence=['#3498db'])
        st.plotly_chart(fig2, use_container_width=True)
        
    with c3:
        st.markdown("**Top Palabras Spam**")
        spam_text = df[df['CLASS'] == 1]['CONTENT']
        cv = CountVectorizer(stop_words='english', max_features=10)
        counts = cv.fit_transform(spam_text)
        words_df = pd.DataFrame({'Palabra': cv.get_feature_names_out(), 'Frecuencia': counts.sum(axis=0).A1})
        fig3 = px.bar(words_df.sort_values('Frecuencia'), x='Frecuencia', y='Palabra', orientation='h')
        st.plotly_chart(fig3, use_container_width=True)

# ==========================================
# PESTAÑA 3: NUBES DE PALABRAS (ASPECTOS)
# ==========================================
with tab3:
    st.subheader("Aspectos más comentados")
    st.write("Las palabras más grandes representan lo que más se repite en cada categoría.")
    
    col_pos, col_neg = st.columns(2)
    
    def generar_nube(data, color):
        texto = " ".join(review for review in data)
        wc = WordCloud(background_color="white", max_words=50, colormap=color).generate(texto)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        return fig

    with col_pos:
        st.success("🌟 Aspectos Positivos")
        pos_data = df[(df['Sentimiento'] == 'Positivo') & (df['CLASS'] == 0)]['CONTENT']
        if not pos_data.empty:
            st.pyplot(generar_nube(pos_data, "Greens"))
        else: st.write("No hay suficientes datos.")

    with col_neg:
        st.error("📉 Aspectos Negativos")
        neg_data = df[(df['Sentimiento'] == 'Negativo') & (df['CLASS'] == 0)]['CONTENT']
        if not neg_data.empty:
            st.pyplot(generar_nube(neg_data, "Reds"))
        else: st.write("No hay suficientes datos.")

# --- FOOTER ---
st.divider()
st.caption("🚀 Sistema avanzado entrenado con Redes Neuronales Multicapa.")
