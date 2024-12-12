import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# Load CIFAR-100 class labels
@st.cache_data
def load_class_names():
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    tar_path = tf.keras.utils.get_file("cifar-100-python.tar.gz", url, extract=True)
    meta_file = os.path.join(os.path.dirname(tar_path), 'meta')
    with open(meta_file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        class_names = [label.decode('utf-8') for label in data[b'fine_label_names']]
    return class_names

# Load the trained model
@st.cache_resource
def load_model():
    model_path = "best_model.keras"  # Ensure this path matches where the model is in your repo
    return tf.keras.models.load_model(model_path)

# Preprocess uploaded image
def preprocess_image(image):
    image = image.resize((32, 32))  # Resize to CIFAR-100 input size
    image = np.array(image).astype('float32') / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app
st.title("CIFAR-100 Image Classifier")
st.write("Upload an image, and the model will predict its class.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    with st.spinner("Classifying..."):
        model = load_model()
        class_names = load_class_names()
        preprocessed_image = preprocess_image(image)
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions) * 100

    # Display results
    st.success("Classification Complete!")
    st.write(f"Predicted Class: **{class_names[predicted_class]}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
