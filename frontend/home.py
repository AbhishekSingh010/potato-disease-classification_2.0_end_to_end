import streamlit as st




    
def homepage():
    st.markdown("""
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
        
    .content-container {
            background-color: #f2f2f2;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        
       .content-container-two {
    background-color: #f2f2f2; /* Light grey color */
    background-image: linear-gradient(to bottom, #f2f2f2, #ffffff); /* White gradient */
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}

        </style>
        """,
        unsafe_allow_html=True)
    # Customizing the welcome message with markdown
    st.markdown(
        """
        <div class=title-text>
        <h1 style="color:#333333;text-align:center;font-family:Arial, sans-serif;
        padding:10px;border-radius:10px;">
        Welcome to the Potato Disease Classification App!</h1>
        </div>
    
        """,
        unsafe_allow_html=True
    )
    
    
    # Features section with different font family
    st.markdown("<h2 style='color:#333333;font-family:Roboto, sans-serif;'>Features:</h2>", unsafe_allow_html=True)
    st.write("- Predict potato disease based on uploaded images.")
    st.write("- Train and evaluate custom models for improved accuracy.")
    st.write("- Get expert advice and solutions through an interactive chatbot.")
    
    # Model Output Classes
    st.markdown(
        """
        
        <h2 style='color:#333333;font-family:Roboto, sans-serif;'>Model Output Classes:</h2>
        <div class=content-container>
        <ul>
        <li style="font-family: 'Courier New', monospace;">- Early Blight</li>
        <li style="font-family: 'Courier New', monospace;">- Late Blight</li>
        <li style="font-family: 'Courier New', monospace;">- Healthy</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Model Architecture
    st.markdown("<h2 style='color:#333333;font-family:Roboto, sans-serif;'>Model Architecture:</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <code style="color:#88888;font-family:Roboto, sans-serif;">
        model=tf.keras.Sequential([  <br>
            resize_and_rescale, <br>
            data_augmentation,  <br>
            tf.keras.layers.Conv2D(32,(3,3),activation='relu', input_shape=input_shape), <br>
            tf.keras.layers.MaxPooling2D((2,2)),<br>
            tf.keras.layers.Conv2D(64, kernel_size=(3,3),activation='relu'),<br>
            tf.keras.layers.MaxPooling2D((2,2)),<br>
            tf.keras.layers.Conv2D(64, kernel_size=(3,3),activation='relu'),<br>
            tf.keras.layers.MaxPooling2D((2,2)),<br>
            tf.keras.layers.Conv2D(64, kernel_size=(3,3),activation='relu'),<br>
            tf.keras.layers.MaxPooling2D((2,2)),<br>
            tf.keras.layers.Conv2D(64, kernel_size=(3,3),activation='relu'),<br>
            tf.keras.layers.MaxPooling2D((2,2)),<br>
            tf.keras.layers.Conv2D(64, kernel_size=(3,3),activation='relu'),<br>
            tf.keras.layers.MaxPooling2D((2,2)),<br>
            tf.keras.layers.Flatten(),<br>
            tf.keras.layers.Dense(64,activation='relu'),<br>
            tf.keras.layers.Dense(n_classes,activation='softmax'),<br>
        ])
        </code>
        <br>
        <div class=content-container-two>
       - Input Layer: The input shape is defined by input_shape, usually representing the dimensions of the input images (e.g., height, width, channels).<br>
       - Convolutional Layers: There are several convolutional layers (Conv2D) with varying numbers of filters (32, 64) and kernel sizes (3x3).<br> Each convolutional layer captures different aspects of the input data.
       - MaxPooling Layers: After each convolutional layer, there is a MaxPooling layer (MaxPooling2D) with a pool size of (2x2), which downsamples the feature maps.<br>
       - Flattening: The Flatten layer converts the 2D output of the last MaxPooling layer into a 1D vector.<br>
       - Dense Layers: There are two dense layers (Dense) with 64 units each, using ReLU activation. <br>The final dense layer has n_classes units with softmax activation, suitable for multi-class classification tasks.
</div>
        """,
        unsafe_allow_html=True
    )
    
    # Chatbot Feature
    st.markdown(
        """
        <h2 style='color:#333333;font-family:Roboto, sans-serif;'>Chatbot Feature:</h2>
        <div class=content-container>
        <p style="font-family: 'Roboto', sans-serif;color:#333333;">The chatbot provides quick answers to user queries related to potato diseases.</p>
        <p style="font-family: 'Roboto', sans-serif;color:#333333;">It offers solutions, advice, and suggestions based on the input query.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # How to Use section
    st.markdown(
        """
        <h2 style='color:#333333;font-family:Roboto, sans-serif;'>How to Use:</h2>
        <div class=content-container>
        <ol>
        <li>Explore the 'About' section to get know more about the project and myself</li>
        <li>Select the 'Prediction App' option from the sidebar to upload potato images for disease classification.</li>
        <li>Explore the 'Model Help' section for  evaluating model performance and choosing best model.</li>
        <li>Explore the'New Model Training ' section for training new cnn model on based of your hyperparameter. </li>
        <li>Explore the 'Chatbot' section to get realtime predication and Engage with the interactive 'Chatbot' to get expert advice and solutions for potato disease queries.</li>
        </ol>
        </div>
        """,
        unsafe_allow_html=True
    )

# if __name__ == "__main__":
#     homepage()
