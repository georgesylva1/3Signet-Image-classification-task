import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("best_model.keras") 

# Define the list of CIFAR-100 class labels (replace with your actual labels)
class_labels = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", 
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", 
    "can", "car", "cat", "cattle", "chair", "chimpanzee", "clock", "cloud", 
    "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin", 
    "elephant", "fox", "girl", "glass", "goldfish", "gorilla", "grapes", 
    "horse", "house", "kangaroo", "keyboard", "kite", "kitten", "lamp", 
    "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", 
    "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", 
    "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", 
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", 
    "ray", "road", "rocket", "rose", "sea", "seal", "shark", "sheep", 
    "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", 
    "sweet_pepper", "table", "tank", "telephone", "television", "tiger", 
    "tractor", "train", "trout", "turtle", "wardrobe", "whale", "willow_tree", 
    "wolf", "woman", "worm"
]

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((32, 32))  # Resize to match CIFAR-100 image size
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Streamlit app
def main():
    st.title("CIFAR-100 Image Classifier")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction)

        # Display the predicted class
        st.write(f"Predicted Class: {class_labels[predicted_class]}")

if __name__ == "__main__":
    main()