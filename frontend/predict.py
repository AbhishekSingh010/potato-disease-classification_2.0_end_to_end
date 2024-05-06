import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os



def predictor():
#  st.write("choose a model you want to use for prediction ")
 models_folder = './models'
 available_models = os.listdir(models_folder)
 selected_model = st.radio("Choose a model for prediction", available_models)
 

 if selected_model:
    model_path = os.path.join(models_folder, selected_model)
 
# Load the pre-trained model
 model = tf.keras.models.load_model(model_path)
 class_names = ["Early Blight", "Late Blight", "Healthy"]

 st.title("Potato Disease Prediction")

 uploaded_file = st.file_uploader("Upload a potato image", type=["jpg", "jpeg", "png"])

 if uploaded_file is not None:
    # Check for valid image format
    if uploaded_file.type not in ['image/jpeg', 'image/png']:
        st.error("Please upload a valid image file (JPG or PNG).")
        # continue

    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image (ensure consistency with FastAPI)
    image_array = np.array(image)  
    img_batch = np.expand_dims(image_array, 0)

    # Make prediction when the 'Predict' button is clicked
    if st.button('Predict'):
        try:
            # Make prediction
            prediction = model.predict(img_batch)
            predicted_class = class_names[np.argmax(prediction[0])]
            confidence = 100 * float(np.max(prediction[0]))

            # Display the prediction result
            st.success(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%")
            return predicted_class      
        except Exception as e:
            st.error(f"Error making prediction: {e}")
          
