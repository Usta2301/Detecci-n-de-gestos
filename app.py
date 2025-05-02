import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
from keras.models import load_model
import platform

# Configuración de página
st.set_page_config(page_title="Detección de Gestos con Teachable Machine", layout="wide")

# Cargar modelo
@st.cache_resource
def load_gesture_model():
    return load_model('keras_model.h5')

model = load_gesture_model()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Encabezado
st.title("🖐️ Aplicación de Detección de Gestos")
st.caption(f"Versión de Python: {platform.python_version()}")

# Modo de entrada
st.sidebar.title("Modo de Entrada")
input_mode = st.sidebar.radio("Selecciona fuente de imagen:", ("📷 Cámara", "📁 Subir imagen"))

# Descripción lateral
with st.sidebar:
    st.markdown("Usa un modelo entrenado con **Teachable Machine** para reconocer gestos con la cámara o imágenes subidas.")
    st.markdown("---")
    if st.checkbox("Mostrar explicación del modelo"):
        st.info("El modelo fue entrenado con imágenes de gestos y exportado como `.h5`. La red espera entradas de tamaño 224x224 con valores normalizados entre -1 y 1.")

# Obtener imagen
img = None
if input_mode == "📷 Cámara":
    img_file_buffer = st.camera_input("Captura una foto")
    if img_file_buffer:
        img = Image.open(img_file_buffer)
else:
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)

# Procesar y predecir si hay imagen
if img:
    st.image(img, caption="Imagen recibida", width=300)

    # Preprocesamiento
    img = ImageOps.fit(img, (224, 224), Image.ANTIALIAS)
    img_array = np.array(img)
    normalized_img = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_img

    # Predicción
    prediction = model.predict(data)[0]

    # Mostrar resultados
    st.subheader("🔍 Resultados de Predicción")
    class_labels = [f"Gesto {i+1}" for i in range(len(prediction))]  # Edita si tienes etiquetas específicas
    results_df = {
        "Gesto": class_labels,
        "Probabilidad": prediction
    }
    st.bar_chart(results_df, use_container_width=True)

    # Mostrar el más probable
    max_index = np.argmax(prediction)
    max_prob = prediction[max_index]
    if max_prob > 0.5:
        st.success(f"✅ Gesto detectado: **{class_labels[max_index]}** con probabilidad {max_prob:.2f}")
    else:
        st.warning("❗ Ningún gesto fue reconocido con alta confianza.")
