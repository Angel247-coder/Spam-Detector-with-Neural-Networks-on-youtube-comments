import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neural_network import MLPClassifier

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="IA Detector de Spam", page_icon="🤖", layout="wide")

# Estilo personalizado para mejorar la estética
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CARGA DE DATOS Y ENTRENAMIENTO (OPTIMIZADO) ---
@st.cache_resource
def preparar_ia():
    # Cargar el dataset
    df = pd.read_csv('Youtube_Unificado_Procesado.csv')
    df = df.dropna(subset=['CONTENT', 'CLASS'])
    df['Etiqueta'] = df['CLASS'].map({0: 'Comentario Real ✅', 1: 'Spam 🚨'})

    # Procesamiento de texto
    vectorizador = TfidfVectorizer(stop_words='english')
    X = vectorizador.fit_transform(df['CONTENT'])
    y = df['CLASS']

    # Entrenar la Red Neuronal (MLP)
    # Usamos random_state para que los resultados sean siempre iguales
    modelo = MLPClassifier(hidden_layer_sizes=(12, 8), max_iter=500, random_state=42)
    modelo.fit(X, y)

    return vectorizador, modelo, df

# Ejecutar la carga
with st.spinner('Cargando cerebro de la IA...'):
    vectorizador, modelo, df = preparar_ia()

# --- 3. ENCABEZADO ---
st.title("🛡️ Sistema de Análisis de Spam en YouTube")
st.write("Esta herramienta utiliza **Deep Learning** para identificar contenido basura y analizar tendencias en los comentarios.")
st.divider()

# --- 4. DISEÑO DE PESTAÑAS ---
tab1, tab2 = st.tabs(["🕵️‍♂️ Analizador en Vivo", "📊 Estadísticas del Dataset"])

# ==========================================
# PESTAÑA 1: ANALIZADOR CON PROBABILIDAD
# ==========================================
with tab1:
    st.subheader("🔍 Análisis de Comentario Individual")
    col_input, col_result = st.columns([2, 1])

    with col_input:
        texto_usuario = st.text_area(
            "Pega aquí el comentario que quieres analizar:",
            placeholder="Ejemplo: Subscribe to my channel for free gift cards!",
            height=150
        )
        boton = st.button("Ejecutar Predicción de IA", type="primary")

    with col_result:
        if boton:
            if texto_usuario.strip() == "":
                st.warning("⚠️ El campo está vacío.")
            else:
                # Transformar y Predecir
                vec = vectorizador.transform([texto_usuario])
                prediccion = modelo.predict(vec)[0]
                probabilidades = modelo.predict_proba(vec)[0]
                confianza = probabilidades[prediccion] * 100

                # Mostrar Resultado
                st.markdown("### Resultado:")
                if prediccion == 1:
                    st.error("🚨 **DETECTADO COMO SPAM**")
                else:
                    st.success("✅ **COMENTARIO LIMPIO**")

                # Mostrar Nivel de Confianza (Tu opción 2)
                st.write(f"**Nivel de confianza:** {confianza:.2f}%")
                st.progress(confianza / 100)
                
                with st.expander("Ver probabilidades detalladas"):
                    st.write(f"Prob. de ser Real: {probabilidades[0]:.4f}")
                    st.write(f"Prob. de ser Spam: {probabilidades[1]:.4f}")

# ==========================================
# PESTAÑA 2: DASHBOARD ESTADÍSTICO
# ==========================================
with tab2:
    st.subheader("📊 Análisis Global del Dataset")
    
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Balance de los Datos")
        fig_pie = px.pie(
            df, names='Etiqueta', 
            color='Etiqueta',
            color_discrete_map={'Comentario Real ✅':'#2ecc71', 'Spam 🚨':'#e74c3c'},
            hole=0.5
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown("#### Palabras clave más usadas en Spam")
        # Filtrar spam y contar palabras
        spam_text = df[df['CLASS'] == 1]['CONTENT']
        cv = CountVectorizer(stop_words='english', max_features=10)
        counts = cv.fit_transform(spam_text)
        
        palabras_df = pd.DataFrame({
            'Palabra': cv.get_feature_names_out(),
            'Frecuencia': counts.sum(axis=0).A1
        }).sort_values(by='Frecuencia', ascending=True)

        fig_bar = px.bar(
            palabras_df, x='Frecuencia', y='Palabra', 
            orientation='h',
            color_discrete_sequence=['#e74c3c']
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# --- FOOTER ---
st.divider()
st.caption("🤖 Modelo: Multi-layer Perceptron (Red Neuronal) | Librerías: Scikit-Learn, Streamlit, Plotly")
