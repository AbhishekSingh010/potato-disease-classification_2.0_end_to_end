
import streamlit as st
from backend.logging_config import logger
from src.dataimporter import dataimporter, class_name
from src.data_preprocessing import preprocess
from src.model_training import model_training
from frontend.predict import predictor









def modeltrainer():
    st.markdown(
        """
        <style>
        .title-text {
            color: #ffffff;
            background-image: linear-gradient(to right, #ff4d4d, #0066ff);
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            font-family: Arial, sans-serif;
            margin-bottom: 20px;
        }
        </style>
         
        """,
        unsafe_allow_html=True)



    
    
    
    
    st.markdown("""
                <div class=title-text>
                New Model training
                </div>
                
               """,unsafe_allow_html=True)
    
    logger.info("main function started in main.py file")
   
    # Data Import
    st.write('Data importing started...')
    dataset = dataimporter()
    classname = class_name(dataset)
    st.write(f'The number of classes in the dataset: {classname}')
   
    # Data Preprocessing
    st.write('Dataset partitioning started...')
    train_ds, val_ds, test_ds = preprocess(dataset=dataset)
    st.write('Dataset partitioning completed successfully.')
    
    
    st.write(" By default:")
    
    st.write("learning rate : 0.001")
    st.write("batch_size : 32")
    st.write("epochs : 10")
    st.write(" To use the default settings, choose 0 below:")
    
    
    

# User Input for Number of Epochs
    epochs = st.number_input("Enter the number of epochs", min_value=0, value=10, step=1)
    if epochs == 0:
        epochs = 10  # Set default epochs if user chooses 0
    st.write(f'The number of epochs you have entered is : {epochs}')

# User Input for Batch Size
    batch_size = st.number_input("Enter the batch size", min_value=0, value=32, step=1)
    if batch_size == 0:
        batch_size = 32  # Set default batch size if user chooses 0
    st.write(f'The batch size you have entered is : {batch_size}')

# User Input for Learning Rate
    learning_rate = st.number_input("Enter the learning rate", min_value=0.0001, value=0.001, step=0.0001)
    if learning_rate == 0.0:
        learning_rate = 0.001  # Set default learning rate if user chooses 0
    st.write(f'The learning rate you have entered is : {learning_rate}')
   
    # Button to Start Model Training
    if st.button("Train New Model"):
        # Model Training
        st.write('Model training started...')
        history = model_training(train_ds, val_ds, test_ds, epochs, batch_size,learning_rate)
        # st.write(f'Model accuracy: {history.summary()}')
        st.write('Model training completed successfully.')
        st.write('Model saved successfully.')

        # Button to Use Trained Model for Prediction
        st.title("Use Your Model for Prediction")
        if st.button("Predict"):
            predictor()
            
        return test_ds    
    
