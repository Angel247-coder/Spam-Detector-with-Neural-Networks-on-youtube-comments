import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neural_network import MLPClassifier

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
# Cambiamos layout="wide" para que los gráficos tengan más espacio
st.set_page_config(page_title="Detector de Spam en YouTube", page_icon="🛡️", layout="wide")

st.title("🛡️ Detector de Spam en YouTube")
st.markdown("""
¡Bienvenido! Esta aplicación utiliza una **Red Neuronal Artificial** para analizar el texto de los comentarios 
y detectar si son **Spam** o **Comentarios Reales**.
""")
st.divider()

# --- 2. ENTRENAMIENTO DEL MODELO Y CARGA DE DATOS ---
@st.cache_resource
def cargar_y_entrenar_modelo():
    df = pd.read_csv('Youtube_Unificado_Procesado.csv')
    df = df.dropna(subset=['CONTENT', 'CLASS'])
    
    # Creamos una columna extra para que los gráficos tengan texto en lugar de 0 y 1
    df['Etiqueta'] = df['CLASS'].map({0: 'Comentario Real', 1: 'Spam'})

    # Entrenar modelo
    X_train, X_test, y_train, y_test = train_test_split(df['CONTENT'], df['CLASS'], test_size=0.3, random_state=42)
    vectorizador = TfidfVectorizer(stop_words='english')
    X_train_vectorizado = vectorizador.fit_transform(X_train)
    red_neuronal = MLPClassifier(hidden_layer_sizes=(8, 8), max_iter=500, random_state=42)
    red_neuronal.fit(X_train_vectorizado, y_train)

    return vectorizador, red_neuronal, df

with st.spinner('Preparando la IA y los datos...'):
    vectorizador, modelo, df = cargar_y_entrenar_modelo()


# --- 3. CREACIÓN DE PESTAÑAS (TABS) ---
tab1, tab2 = st.tabs(["🕵️‍♂️ Analizador en Vivo", "📊 Panel de Estadísticas"])

# === PESTAÑA 1: EL ANALIZADOR ===
with tab1:
    st.subheader("🕵️‍♂️ Pruébalo tú mismo")
    st.write("Escribe o pega un comentario de YouTube abajo:")

    comentario_usuario = st.text_area("Comentario:", placeholder="Ejemplo: OMG check out my new channel!!!", key="text_area_1")

    if st.button("Analizar Comentario", type="primary"):
        if comentario_usuario.strip() == "":
            st.warning("⚠️ Por favor, escribe un comentario primero.")
        else:
            mensaje_vectorizado = vectorizador.transform([comentario_usuario])
            prediccion = modelo.predict(mensaje_vectorizado)[0]

            if prediccion == 1:
                st.error("🚨 **¡ALERTA DE SPAM!** Este comentario parece contenido basura o promocional.")
            else:
                st.success("✅ **¡COMENTARIO LIMPIO!** Parece ser un usuario real comentando de forma genuina.")

# === PESTAÑA 2: EL PANEL DE ESTADÍSTICAS ===
with tab2:
    st.subheader("📊 Análisis del Dataset de Entrenamiento")
    st.write("Explora los datos con los que la red neuronal ha sido entrenada.")
    
    # Dividimos la pantalla en dos columnas para poner los gráficos lado a lado
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Proporción de Spam vs Comentarios Reales**")
        # Creamos un gráfico de dona (pie chart con un agujero) usando Plotly
        fig_pie = px.pie(df, names='Etiqueta', color='Etiqueta',
                         color_discrete_map={'Comentario Real':'#28a745', 'Spam':'#dc3545'},
                         hole=0.4)
        # Hacemos que la leyenda esté debajo para ahorrar espacio
        fig_pie.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown("**Top 10 palabras más usadas por los Spammers**")
        # Filtramos solo los comentarios que son Spam
        spam_df = df[df['CLASS'] == 1]
        
        # Usamos CountVectorizer para contar las palabras más frecuentes, ignorando palabras comunes (stop words)
        contador = CountVectorizer(stop_words='english', max_features=10)
        spam_palabras = contador.fit_transform(spam_df['CONTENT'])
        
        # Sumamos y preparamos los datos
        frecuencias = spam_palabras.sum(axis=0).A1
        nombres_palabras = contador.get_feature_names_out()
        df_palabras = pd.DataFrame({'Palabra': nombres_palabras, 'Frecuencia': frecuencias})
        df_palabras = df_palabras.sort_values(by='Frecuencia', ascending=True) # Ascendente para el gráfico de barras horizontal
        
        # Creamos un gráfico de barras horizontales
        fig_bar = px.bar(df_palabras, x='Frecuencia', y='Palabra', orientation='h',
                         color_discrete_sequence=['#dc3545'])
        st.plotly_chart(fig_bar, use_container_width=True)

# --- FOOTER ---
st.divider()
st.caption("Desarrollado con Streamlit, Scikit-Learn y Plotly 🚀")
