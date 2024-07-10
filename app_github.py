import streamlit as st
import joblib
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
from io import BytesIO

# Define the image size for the model
IMG_SIZE = (224, 224)

# Define the paths to the models
FEATURE_EXTRACTOR_PATH = 'mobilenetv2.h5'
SVM_MODEL_PATH = 'best_model.joblib'

@st.cache_resource
def load_models():
    try:
        feature_extractor = tf.keras.models.load_model(FEATURE_EXTRACTOR_PATH)
    except Exception as e:
        st.error(f"Error loading feature extractor model: {e}")
        feature_extractor = None

    try:
        svm_model = joblib.load(SVM_MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading SVM model: {e}")
        svm_model = None

    return feature_extractor, svm_model

feature_extractor, svm_model = load_models()

def preprocess_image(image):
    """
    Preprocess the input image for prediction.

    Args:
        image (PIL.Image): The input image.

    Returns:
        np.ndarray: The preprocessed image as a numpy array.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img = image.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array

def predict_image(image_array):
    """
    Predict the class of the input image using the SVM model.

    Args:
        image_array (np.ndarray): The preprocessed image array.

    Returns:
        str: The predicted class label.
    """
    if feature_extractor and svm_model:
        features = feature_extractor.predict(image_array)
        features = features.reshape((1, -1))
        prediction = svm_model.predict(features)
        class_labels = ['Glass Bottle', 'Plastic Bottle', 'Tin Can']
        return class_labels[prediction[0]]
    else:
        return "Error: Model not loaded properly."

def main():
    st.title("Object Classification")

    input_method = st.selectbox("Choose Image Input Method", ("Please Select", "Upload Image", "Predict from URL"))

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            img_array = preprocess_image(image)

            if st.button("Predict"):
                predicted_class = predict_image(img_array)
                st.success(f"Predicted Class: {predicted_class}")

    elif input_method == "Predict from URL":
        url = st.text_input("Enter Image URL")

        if url:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    st.image(image, caption="Image from URL", use_column_width=True)
                    img_array = preprocess_image(image)

                    if st.button("Predict"):
                        predicted_class = predict_image(img_array)
                        st.success(f"Predicted Class: {predicted_class}")
                else:
                    st.error(f"Error: Unable to fetch image from URL. Status code: {response.status_code}")

            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
