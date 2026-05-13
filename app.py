import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
# Esto le da un título a la pestaña del navegador y un diseño más centrado
st.set_page_config(page_title="Detector de Spam en YouTube", page_icon="🛡️", layout="centered")

# --- 2. ENCABEZADO Y DISEÑO ---
st.title("🛡️ Detector de Spam en YouTube")
st.markdown("""
¡Bienvenido! Esta aplicación utiliza una **Red Neuronal Artificial** para analizar el texto de los comentarios 
y detectar si son **Spam** (publicidad no deseada, links peligrosos, etc.) o **Comentarios Reales**.
""")
st.divider() # Línea separadora visual

# --- 3. ENTRENAMIENTO DEL MODELO (EN CACHÉ) ---
# st.cache_resource hace que este bloque solo se ejecute UNA vez. 
# Así la web carga súper rápido las siguientes veces.
@st.cache_resource
def cargar_y_entrenar_modelo():
    # Cargar datos
    df = pd.read_csv('Youtube_Unificado_Procesado.csv')
    df = df.dropna(subset=['CONTENT', 'CLASS'])

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(df['CONTENT'], df['CLASS'], test_size=0.3, random_state=42)

    # Vectorizar texto
    vectorizador = TfidfVectorizer(stop_words='english')
    X_train_vectorizado = vectorizador.fit_transform(X_train)

    # Entrenar Red Neuronal
    red_neuronal = MLPClassifier(hidden_layer_sizes=(8, 8), max_iter=500, random_state=42)
    red_neuronal.fit(X_train_vectorizado, y_train)

    return vectorizador, red_neuronal

# Mostramos un pequeño mensaje de carga mientras la IA "estudia"
with st.spinner('Entrenando la Inteligencia Artificial... (Esto solo tarda unos segundos)'):
    vectorizador, modelo = cargar_y_entrenar_modelo()


# --- 4. INTERFAZ INTERACTIVA PARA EL USUARIO ---
st.subheader("🕵️‍♂️ Pruébalo tú mismo")
st.write("Escribe o pega un comentario de YouTube abajo y la IA te dirá qué opina:")

# Caja de texto para que el usuario escriba
comentario_usuario = st.text_area(
    "Comentario:", 
    placeholder="Ejemplo: OMG check out my new channel and subscribe to win a free iPhone!!!"
)

# Botón principal para analizar
if st.button("Analizar Comentario", type="primary"):
    
    # Validamos que no esté vacío
    if comentario_usuario.strip() == "":
        st.warning("⚠️ Por favor, escribe un comentario primero.")
    else:
        # Predecimos con el modelo
        mensaje_vectorizado = vectorizador.transform([comentario_usuario])
        prediccion = modelo.predict(mensaje_vectorizado)[0]

        # Mostramos el resultado con colores y alertas visuales
        if prediccion == 1:
            st.error("🚨 **¡ALERTA DE SPAM!** Este comentario parece contenido basura o promocional.")
        else:
            st.success("✅ **¡COMENTARIO LIMPIO!** Parece ser un usuario real comentando de forma genuina.")

# --- FOOTER ---
st.divider()
st.caption("Desarrollado con Streamlit y Scikit-Learn 🚀")
