import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Function to load the Keras model
@st.cache(allow_output_mutation=True)
def load_keras_model():
    try:
        model = load_model('c:\Document\cifar10_model_new.h5')  # Update the filepath if necessary
        return model
    except OSError as e:
        st.error(f"Error loading Keras model: {e}")
        return None

# Load the trained model
model = load_keras_model()

# Class names for CIFAR-10 dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Streamlit app title
st.title('CIFAR-10 Image Classification')

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = image.resize((32, 32))
    image = img_to_array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    # Make a prediction if the model is loaded successfully
    if model:
        prediction = model.predict(image)
        predicted_class = class_names[np.argmax(prediction)]
        st.write(f'Predicted Class: {predicted_class}')