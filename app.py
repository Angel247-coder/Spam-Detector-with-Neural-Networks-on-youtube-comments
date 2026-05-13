import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neural_network import MLPClassifier
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="IA YouTube Global Analyzer", page_icon="🌎", layout="wide")

# --- 2. CARGA DE MODELOS ---
@st.cache_resource
def cargar_modelos_pro():
    # A. Cargar Datos
    df = pd.read_csv('Youtube_Unificado_Procesado.csv')
    df = df.dropna(subset=['CONTENT', 'CLASS'])
    df['Etiqueta'] = df['CLASS'].map({0: 'Real ✅', 1: 'Spam 🚨'})

    # B. IA de Sentimientos Multilingüe (Hugging Face)
    # Este modelo entiende Español, Inglés, Francés, Alemán, etc.
    modelo_sent = pipeline(
        "sentiment-analysis", 
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )

    # C. IA de Spam (Tu Red Neuronal)
    vectorizador = TfidfVectorizer(stop_words='english', max_features=2000)
    X = vectorizador.fit_transform(df['CONTENT'])
    y = df['CLASS']
    modelo_spam = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
    modelo_spam.fit(X, y)

    return vectorizador, modelo_spam, df, modelo_sent

with st.spinner('Iniciando motores de IA Multilingüe... (Esto puede tardar 1 min la primera vez)'):
    vectorizador, modelo_spam, df, sentiment_pipe = cargar_modelos_pro()

# --- 3. LÓGICA ---
def analizar_comentario(texto):
    # Spam
    vec = vectorizador.transform([texto])
    es_spam = modelo_spam.predict(vec)[0]
    probs = modelo_spam.predict_proba(vec)[0]
    
    # Sentimiento (Transformer)
    res_sent = sentiment_pipe(texto)[0]
    # El modelo devuelve: 'positive', 'negative' o 'neutral'
    sent_label = res_sent['label'].capitalize()
    
    return es_spam, probs[es_spam] * 100, sent_label

# --- 4. INTERFAZ ---
st.title("🌎 IA YouTube Sentiment & Spam Analyzer (Bilingüe)")
st.write("Detección de Spam con Redes Neuronales y Sentimiento con Transformers Multilingües.")
st.divider()

t1, t2, t3 = st.tabs(["🕵️‍♂️ Analizador Bilingüe", "📊 Dashboard", "☁️ Nubes de Palabras"])

with t1:
    col1, col2 = st.columns([2, 1])
    with col1:
        user_input = st.text_area("Escribe un comentario (Español o Inglés):", placeholder="Ej: This video is awesome! / ¡Qué buen tutorial!")
        btn = st.button("Analizar con Deep Learning", type="primary")

    with col2:
        if btn and user_input.strip() != "":
            spam, confianza, sentimiento = analizar_comentario(user_input)
            
            st.subheader("Resultado:")
            if spam == 1:
                st.error(f"🚨 SPAM ({confianza:.1f}%)")
            else:
                st.success(f"✅ REAL ({confianza:.1f}%)")
            
            # Iconos dinámicos según sentimiento
            icons = {"Positive": "🌟 Positivo", "Negative": "💢 Negativo", "Neutral": "😶 Neutral"}
            st.info(f"**Sentimiento detectado:** {icons.get(sentimiento, sentimiento)}")
            st.progress(confianza/100)

with t2:
    st.subheader("Estadísticas del Dataset")
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig_spam = px.pie(df, names='Etiqueta', title="Proporción de Spam", color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig_spam, use_container_width=True)
    
    with col_b:
        st.write("**Análisis de Sentimiento Masivo**")
        st.info("Para analizar el sentimiento de miles de filas se requiere mucha potencia. Usa el analizador individual para probar la precisión.")

with t3:
    st.subheader("Aspectos más comentados")
    # Filtramos por palabras clave de spam para mostrar qué dicen los bots
    spam_text = " ".join(df[df['CLASS'] == 1]['CONTENT'])
    real_text = " ".join(df[df['CLASS'] == 0]['CONTENT'])
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("☁️ Palabras en Spam")
        wc_spam = WordCloud(background_color="white", colormap="Reds").generate(spam_text)
        fig, ax = plt.subplots()
        ax.imshow(wc_spam)
        ax.axis("off")
        st.pyplot(fig)
    
    with c2:
        st.write("☁️ Palabras en Reales")
        wc_real = WordCloud(background_color="white", colormap="Blues").generate(real_text)
        fig, ax = plt.subplots()
        ax.imshow(wc_real)
        ax.axis("off")
        st.pyplot(fig)
