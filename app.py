import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from transformers import pipeline
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
from itertools import islice
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Auditor de YouTube IA", page_icon="🎬", layout="wide")

@st.cache_resource
def cargar_modelos():
    # Carga de dataset base para entrenar el detector de Spam
    df_base = pd.read_csv('Youtube_Unificado_Procesado.csv').dropna(subset=['CONTENT', 'CLASS'])
    
    # Modelo Spam (Red Neuronal)
    vec = TfidfVectorizer(stop_words='english', max_features=2000)
    X = vec.fit_transform(df_base['CONTENT'])
    m_spam = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
    m_spam.fit(X, df_base['CLASS'])
    
    # Modelo Sentimiento (Transformer Multilingüe)
    m_sent = pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    
    return vec, m_spam, m_sent

vec, m_spam, m_sent = cargar_modelos()

# --- 2. LÓGICA DE EXTRACCIÓN ---
def extraer_comentarios(url, cantidad):
    downloader = YoutubeCommentDownloader()
    try:
        comments = downloader.get_comments_from_url(url, sort_by=SORT_BY_RECENT)
        # Tomamos solo N comentarios para no saturar el servidor
        datos = []
        for c in islice(comments, cantidad):
            datos.append({
                'Autor': c['author'],
                'Comentario': c['text'],
                'Fecha': c['time']
            })
        return pd.DataFrame(datos)
    except Exception as e:
        st.error(f"Error al conectar con YouTube: {e}")
        return None

# --- 3. INTERFAZ ---
st.title("🎬 Auditor de Audiencia YouTube Pro")
st.markdown("Analiza la salud de cualquier video mediante **Inteligencia Artificial**.")
st.divider()

# Barra lateral para configuración
with st.sidebar:
    st.header("⚙️ Ajustes de Auditoría")
    url_video = st.text_input("URL del Video", placeholder="https://www.youtube.com/watch?v=...")
    n_comments = st.slider("Comentarios a analizar", 10, 200, 50)
    procesar = st.button("🚀 Iniciar Auditoría", type="primary")

if procesar:
    if not url_video:
        st.warning("⚠️ Por favor, introduce una URL de YouTube.")
    else:
        with st.spinner(f"Analizando los últimos {n_comments} comentarios..."):
            # A. Descarga
            df_audit = extraer_comentarios(url_video, n_comments)
            
            if df_audit is not None and not df_audit.empty:
                # B. Análisis IA
                resultados = []
                for msg in df_audit['Comentario']:
                    # Spam Check
                    v = vec.transform([msg])
                    is_spam = m_spam.predict(v)[0]
                    conf = m_spam.predict_proba(v)[0][is_spam] * 100
                    
                    # Sentiment Check
                    sent_res = m_sent(msg)[0]
                    
                    resultados.append({
                        'Spam': "🚨 SÍ" if is_spam == 1 else "✅ NO",
                        'Confianza': f"{conf:.1f}%",
                        'Sentimiento': sent_res['label'].capitalize()
                    })
                
                df_final = pd.concat([df_audit, pd.DataFrame(resultados)], axis=1)

                # --- C. DASHBOARD DE RESULTADOS ---
                c1, c2, c3 = st.columns(3)
                total = len(df_final)
                spam_count = (df_final['Spam'] == "🚨 SÍ").sum()
                pos_count = (df_final['Sentimiento'] == "Positive").sum()

                c1.metric("Comentarios Extraídos", total)
                c2.metric("Nivel de Spam", f"{(spam_count/total)*100:.1f}%", delta=f"{spam_count} bots", delta_color="inverse")
                c3.metric("Felicidad Audiencia", f"{(pos_count/total)*100:.1f}%")

                st.divider()

                # Visualización Gráfica
                col_a, col_b = st.columns(2)
                with col_a:
                    fig_sent = px.pie(df_final, names='Sentimiento', title="Clima Emocional", 
                                     color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c', 'Neutral':'#95a5a6'})
                    st.plotly_chart(fig_sent, use_container_width=True)
                
                with col_b:
                    st.write("**Nube de Temas Reales** (Sin Spam)")
                    reales = " ".join(df_final[df_final['Spam'] == "✅ NO"]['Comentario'])
                    if reales.strip():
                        wc = WordCloud(background_color="white", colormap="Blues").generate(reales)
                        fig_wc, ax = plt.subplots(); ax.imshow(wc); ax.axis("off")
                        st.pyplot(fig_wc)

                # Tabla detallada
                st.subheader("📋 Detalle de la Auditoría")
                st.dataframe(df_final, use_container_width=True)
                
                # Botón de Descarga
                csv = df_final.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Descargar Reporte CSV", data=csv, file_name="auditoria_youtube.csv")

            else:
                st.error("No se pudieron obtener comentarios. Verifica que el video no tenga los comentarios desactivados.")
else:
    st.info("Pega una URL de YouTube en la barra lateral para comenzar el análisis en tiempo real.")
