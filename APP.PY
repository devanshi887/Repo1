import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
# Page config
st.set_page_config(page_title="Veg & Fruit Classifier", layout="wide")

# Dark Theme
st.markdown("""
<style>
.stApp {
    background-color:#0e1117;
    color:white;
}
h1{
    text-align:center;
    color:#00ffaa;
}
</style>
""", unsafe_allow_html=True)

st.title("🥦 Veg & Fruit Image Classifier")

# Load Model
model = tf.keras.models.load_model("veg_fruit (5).keras")
# Load class names
with open("class_names.json") as f:
    class_names = json.load(f)["class_names"]

# Class Names
class_names = [
'apple','banana','beetroot','bell pepper','cabbage','capsicum','carrot',
'cauliflower','chilli pepper','corn','cucumber','eggplant','garlic',
'ginger','grapes','jalepeno','kiwi','lemon','lettuce','mango','onion',
'orange','paprika','pear','peas','pineapple','pomegranate','potato',
'raddish','soy beans','spinach','sweetcorn','sweetpotato','tomato',
'turnip','watermelon'
]

uploaded_file = st.file_uploader("📷 Upload Veg or Fruit Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # Side by Side Layout
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]

    predicted_class = np.argmax(prediction)
    predicted_name = class_names[predicted_class]

    confidence = prediction[predicted_class] * 100

    with col2:
        st.subheader("Prediction Result")
        st.success(f"🥗 {predicted_name}")

        st.write(f"Confidence: **{confidence:.2f}%**")

        st.progress(float(confidence/100))