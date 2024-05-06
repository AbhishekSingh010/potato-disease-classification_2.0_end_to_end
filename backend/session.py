
import streamlit as st

# Define a SessionState class to hold values across sessions
# Initialize session state
if 'my_list' not in st.session_state:
    st.session_state.my_list = {}

def listing(response):
    if st.session_state.my_list is None:
        st.session_state.my_list = {}  # Initialize my_list as an empty dictionary

    if len(st.session_state.my_list) >= 0 and len(st.session_state.my_list) <= 3:
        index = len(st.session_state.my_list) + 1
        st.session_state.my_list[index] = response
    elif len(st.session_state.my_list) >= 3:
        st.session_state.my_list = {}  # Clear session state if list has 2 or more elements

    return st.session_state.my_list

