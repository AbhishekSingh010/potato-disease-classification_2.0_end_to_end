
from backend.logging_config import logger
from frontend.chat_bot import chatbot
from frontend.about import about_page
from backend.session import listing



from frontend.home import homepage
from frontend.predict import predictor
from frontend.model_trainer import modeltrainer
from frontend.monitor import monitor
import streamlit as st






logger.info("main function started in main.py file")
   
#    1.data import  =>dataimporter.py =>load the data and give class name too
#    2.data augmentation and rescaling =>data preprocessing.py,=>train test split
#    3.modeltraining =>summary get logged
#    4.src/evaluation.py=>evaluating model accuracy =>chossing best model=>logging the model accuracy confusion metrics etc
#    5.model monitoring and accuracy matrics ->monitor.py frontend
#    try to monitor the model and developing all sort of comparisons and about model data on some streamlit page 
#    model monitoring step
#    6.predicted page => we basically predict the output
#    7.gemini chatbot for => end to end communication and user quaries solution or q/a bot
   

response_=''
import streamlit as st

# Define custom CSS styles
st.markdown(
    """
    <style>
    .title-text {
        color: #003366;  /* Dark blue color for title */
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .sidebar {
        background-color: #f0f0f0;  /* Light gray background for sidebar */
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar-item {
        margin-bottom: 10px;
    }
    .content {
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def chatbot_():
    col1, col2 = st.columns(2)
    with col1:
        response = predictor()
        
        if str(response) != "NULL":
            my_list = listing(response)
        if my_list:  # Check if my_list is not empty
            st.write(my_list[1])  # Display the first response in the dictionary
        st.write(my_list)
        print(my_list)
    with col2:
        # st.write("new one", my_list)
        if my_list:  # Check if my_list is not empty
            chatbot(my_list)

def home():
    st.markdown("<h1 class='title-text'>Potato Disease Classification App</h1>", unsafe_allow_html=True)
   

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose an option",
        ["Home", "Prediction app", "About", "Model Help", "New Model Training", "Chatbot"]
    )

    if app_mode == "Home":
        logger.info("hompage get called")
        homepage()
        st.write("This is the home page content.")
    elif app_mode == "Prediction app":
        st.subheader("This is the prediction app content")
        logger.info("hompage get called")
        predictor()
        st.write("This page allows users to upload potato images for disease classification.")
    elif app_mode == "About":
        logger.info("about_page get called")
        about_page()
        st.write("This page provides information about the app and its development.")
    elif app_mode == "Model Help":
        logger.info("model-help monitor get called")
        monitor()
        st.subheader("Model Help Page")
        st.write("This page provides information about the app's models and their performance.")
    elif app_mode == "New Model Training":
        logger.info("model trainer get  called")
        modeltrainer()
    elif app_mode == "Chatbot":
        st.subheader("Chatbot Page")
        logger.info("chatbot get called")
        chatbot_()
        st.write("This page provides an interface for interacting with the chatbot.")

if __name__ == "__main__":
    home()
