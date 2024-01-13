import streamlit as st
import zipfile
import numpy as np
import tensorflow as tf
from PIL import Image

# Extract the CNN model from the zip file
cnn_path = "/content/gdrive/My Drive/cnn.zip"
with zipfile.ZipFile(cnn_path, 'r') as zip_ref:
    zip_ref.extractall('')

# Load the CNN model
cnn = tf.keras.models.load_model('cnn')

# Define the Streamlit app
st.title("Medical Imaging")


# Function to process and classify the uploaded image
def process_image(image):
    # Convert the image to RGB mode
    image = image.convert('RGB')

    # Resize the image
    image = image.resize((64, 64))

    # Convert the image to an array and normalize
    img_array = np.array(image).astype('float32')
    img_array = img_array / 255.0

    # Expand dimensions to match the expected input shape
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# Display the image upload widget
uploaded_file = st.file_uploader("Select Image", type=["png", "jpg", "jpeg"])

# Perform prediction and display the result
if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)

    # Process and classify the image
    processed_image = process_image(image)
    prediction = cnn.predict(processed_image)
    prediction_label = (prediction > 0.5).astype(int)
    diagnosis = "Pneumonia" if prediction_label else "No Pneumonia"

    # Display the processed image
    st.image(image)

    # Display the predicted diagnosis and confidence score
    confidence = prediction[0][0] if prediction_label else 1 - prediction[0][0]
    st.header("Prediction")
    st.subheader("Diagnosis")
    st.write(diagnosis)
    st.subheader("Confidence Score")
    st.write(f"{confidence * 100:.2f}%")

    # Get the output of intermediate layers
    intermediate_layer_model = tf.keras.models.Model(inputs=cnn.input, outputs=cnn.layers[3].output)
    intermediate_output = intermediate_layer_model.predict(processed_image)

# Display the model architecture
st.subheader("Model Architecture")
st.text(cnn.summary())
