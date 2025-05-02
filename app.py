import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
from keras.models import load_model
import platform

# Configuración de la página
st.set_page_config(
    page_title="Detección de Gestos con Teachable Machine",
    layout="wide"
)

# Carga y cache del modelo
@st.cache_resource
def load_gesture_model():
    return load_model('keras_model.h5')

model = load_gesture_model()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Encabezado
st.title("🖐️ Aplicación de Detección de Gestos")
st.caption(f"Versión de Python: {platform.python_version()}")

# Sidebar: selección de modo de entrada
st.sidebar.title("Modo de Entrada")
input_mode = st.sidebar.radio("Selecciona fuente de imagen:", ("📷 Cámara", "📁 Subir imagen"))

# Sidebar: explicación del modelo
with st.sidebar:
    if st.checkbox("Mostrar explicación del modelo"):
        st.info(
            "Este modelo fue entrenado en Teachable Machine y exportado como `.h5`. "
            "Es una red que espera entradas de 224×224 px normalizadas en [−1, 1]."
        )

# Obtener la imagen
img = None
if input_mode == "📷 Cámara":
    buf = st.camera_input("Captura una foto")
    if buf:
        img = Image.open(buf)
else:
    uploaded = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)

# Procesar y predecir
if img:
    st.image(img, caption="Imagen recibida", width=300)

    # --------- Preprocesamiento compatible Pillow ≥10 y <10 ---------
    # Seleccionar método de remuestreo
    if hasattr(Image, "Resampling"):
        resample_method = Image.Resampling.LANCZOS
    else:
        resample_method = Image.ANTIALIAS

    # Redimensionar y recortar manteniendo aspecto
    img = ImageOps.fit(img, (224, 224), method=resample_method)
    # -----------------------------------------------------------------

    # Convertir a array y normalizar entre −1 y 1
    arr = np.array(img).astype(np.float32)
    norm = (arr / 127.0) - 1
    data[0] = norm

    # Inferencia
    prediction = model.predict(data)[0]

    # Mostrar resultados
    st.subheader("🔍 Resultados de Predicción")
    # Asume tantas etiquetas como salidas del modelo
    class_labels = [f"Gesto {i+1}" for i in range(len(prediction))]
    result_df = {
        "Gesto": class_labels,
        "Probabilidad": prediction
    }
    st.bar_chart(result_df, use_container_width=True)

    # Gestor de umbral
    max_idx = np.argmax(prediction)
    max_prob = prediction[max_idx]
    if max_prob > 0.5:
        st.success(f"✅ Gesto detectado: **{class_labels[max_idx]}** ({max_prob:.2f})")
    else:
        st.warning("❗ Ningún gesto reconocido con probabilidad suficiente.")
else:
    st.info("Seleccione una imagen desde la barra lateral para comenzar.")
