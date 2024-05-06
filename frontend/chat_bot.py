import streamlit as st

from chat.chat_back import content_creation


def chatbot(my_dict):
    if my_dict is not None: #and len(my_dict) >= 3:
        for key in my_dict:
            if my_dict[key] in ["Early Blight", "Late Blight", "Healthy"]:  # Check the value associated with the key
                predict_output = my_dict.get(key, '')
                break  # Exit the loop after finding the first valid key
        else:
            predict_output = ''  # Set predict_output to '' if no valid key is found
    else:
        predict_output = '' 
   

    # Define CSS styles for chat messages
    chat_container_style = """
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
        background-color: #f9f9f9;
        margin-bottom: 10px;
    """
    user_message_style = "color: #006eff; font-weight: bold;"
    bot_message_style = "color: #00a85e; font-weight: bold;"

    st.title("Interactive Chatbot")

    # User input fields
    user_quary = st.text_input("Enter your query:")

  
    
    with st.container():
        st.markdown("<h2 style='margin-bottom: 20px;'>Chat Messages</h2>", unsafe_allow_html=True)
        chat_messages = []

        # Function to display user message
        def display_user_message(message):
            st.markdown(f"<div style='{chat_container_style + user_message_style}'>{message}</div>", unsafe_allow_html=True)

        # Function to display bot message
        def display_bot_message(message):
            st.markdown(f"<div style='{chat_container_style + bot_message_style}'>{message}</div>", unsafe_allow_html=True)

        # Submit button to trigger chatbot response
        if st.button("Submit"):
            display_user_message(f"User: {user_quary}")

            # Call chatbot backend function and display response
            # st.write(predict_output)
            response = content_creation(user_quary,predict_output)
            display_bot_message(f"Bot: {response}")





