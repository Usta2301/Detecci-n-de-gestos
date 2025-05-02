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

# Diccionario de descripciones para cada gesto
# Ajusta estas entradas según las clases reales de tu modelo
gesture_descriptions = {
    "Gesto 1": "Encogimiento de hombros: transmite duda o indiferencia.",
    "Gesto 2": "Pulgar arriba: indica aprobación o acuerdo.",
    "Gesto 3": "Pulgar abajo: indica desaprobación o desacuerdo.",
    "Gesto 4": "Señal de victoria (V): suele transmitir celebración o paz.",
    # añade más según necesites…
}

# Encabezado
st.title("🖐️ Aplicación de Detección de Gestos")
st.caption(f"Versión de Python: {platform.python_version()}")

# Sidebar: modo de entrada
st.sidebar.title("Modo de Entrada")
input_mode = st.sidebar.radio("Selecciona fuente de imagen:", ("📷 Cámara", "📁 Subir imagen"))

# Sidebar: explicación del modelo
with st.sidebar:
    if st.checkbox("Mostrar explicación del modelo"):
        st.info(
            "Modelo entrenado en Teachable Machine, exportado como `.h5`. "
            "Recibe imágenes de 224×224 px, normalizadas en [−1, 1]."
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

    # --------- Preprocesamiento Pillow ≥10 / <10 ---------
    if hasattr(Image, "Resampling"):
        resample_method = Image.Resampling.LANCZOS
    else:
        resample_method = Image.ANTIALIAS
    img = ImageOps.fit(img, (224, 224), method=resample_method)
    # -----------------------------------------------------------------

    # Convertir a array y normalizar
    arr = np.array(img).astype(np.float32)
    norm = (arr / 127.0) - 1
    data[0] = norm

    # Inferencia
    prediction = model.predict(data)[0]

    # Labels dinámicos
    class_labels = [f"Gesto {i+1}" for i in range(len(prediction))]

    # Mostrar barra de probabilidades
    st.subheader("🔍 Probabilidades por Gesto")
    df_chart = {
        "Gesto": class_labels,
        "Probabilidad": prediction
    }
    st.bar_chart(df_chart, use_container_width=True)

    # Umbral para considerar gesto detectado
    threshold = 0.5
    detected = [(class_labels[i], prediction[i]) for i in range(len(prediction)) if prediction[i] > threshold]

    if detected:
        for name, prob in detected:
            desc = gesture_descriptions.get(name, "Descripción no disponible.")
            st.success(f"✅ **{name}** ({prob:.2f}) — {desc}")
    else:
        st.warning("❗ Ningún gesto reconocido con probabilidad suficiente.")

else:
    st.info("Selecciona o captura una imagen para comenzar.")
