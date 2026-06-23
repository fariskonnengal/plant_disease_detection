import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from groq import Groq

@st.cache_resource(show_spinner=False)
def load_disease_model():
    return load_model('plant_disease_model.keras')

model = load_disease_model()

# Groq client for the AI Doctor feature (free LLM)
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

@st.cache_data(show_spinner=False)
def get_ai_doctor_explanation(disease_name):
    """Ask the LLM to explain the detected condition in farmer-friendly terms."""
    readable_name = disease_name.replace('___', ' - ').replace('_', ' ')

    if 'healthy' in disease_name.lower():
        prompt = (
            f"A plant disease detection model scanned a leaf and classified it as: "
            f"\"{readable_name}\". This means the plant appears healthy.\n\n"
            "In simple, friendly language for a farmer with no medical/scientific background:\n"
            "- Briefly confirm the plant looks healthy\n"
            "- Give 3 short practical tips to keep it that way (watering, sunlight, pests, etc.)\n"
            "Keep it short, use bullet points, no markdown headers."
        )
    else:
        prompt = (
            f"You are an AI plant doctor. A plant disease detection model scanned a leaf "
            f"image and classified it as: \"{readable_name}\".\n\n"
            "Explain this to a farmer with no scientific background, in simple plain language. "
            "Cover, briefly and as bullet points:\n"
            "- What this disease is (1-2 lines)\n"
            "- Common symptoms to look for\n"
            "- What usually causes it\n"
            "- How to treat/control it (practical, actionable steps)\n"
            "- How to prevent it in the future\n\n"
            "Keep the whole answer concise (under 200 words), no markdown headers, just bullet points."
        )

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=500,
    )
    return response.choices[0].message.content


def get_gradcam_heatmap(model, img_array, last_conv_layer_name='Conv_1'):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

st.title("Plant Disease Detection")
st.write("Upload a leaf image to detect the disease")

uploaded_file = st.file_uploader("Choose a leaf image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))

    st.image(img_resized, caption='Uploaded Image', width=300)

    img_array = np.expand_dims(img_resized, axis=0) / 255.0
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions[0])
    confidence = predictions[0][pred_index] * 100

    st.success(f"Predicted Disease: {class_names[pred_index]}")
    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("🩺 AI Plant Doctor")
    with st.spinner("Consulting the AI doctor..."):
        try:
            explanation = get_ai_doctor_explanation(class_names[pred_index])
            st.markdown(explanation)
        except Exception as e:
            st.error(f"AI Doctor is unavailable right now: {e}")

    heatmap = get_gradcam_heatmap(model, img_array)

    if np.max(heatmap) > 0:
        heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        heatmap = np.clip(heatmap, 0, 1)
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(img_resized, 0.6, heatmap_colored, 0.4, 0)

        st.subheader("Grad-CAM Visualization")
        col1, col2 = st.columns(2)
        with col1:
            st.image(heatmap_colored, caption='Heatmap')
        with col2:
            st.image(superimposed, caption='Superimposed')