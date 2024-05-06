import streamlit as st

def about_page():
    st.title("About Potato Disease Prediction Project")

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
        .content-container {
            background-color: #f2f2f2;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .contact-info {
            font-weight: bold;
        }
        .team-member {
            margin-bottom: 10px;
        }
        .logo {
            max-width: 100px;
            margin-right: 10px;
            vertical-align: middle;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="content-container">
        <h2>Project Description</h2>
        <p>
        The Potato Disease Prediction Project is a machine learning-based application that aims to predict and classify potato diseases accurately using deep learning techniques.
        </p>

        <h2>Features</h2>
        <ul>
        <li>Predict potato disease based on uploaded images.</li>
        <li>Train and evaluate custom models for improved accuracy.</li>
        <li>Get expert advice and solutions through an interactive chatbot.</li>
        <li>Visualize disease patterns and trends with insightful graphs.</li>
        <li>Receive real-time alerts and recommendations for disease prevention.</li>
        </ul>

        <h2>Technology Stack</h2>
        <p>
        The project utilizes state-of-the-art technologies including TensorFlow for deep learning, Streamlit for the web interface, and Python for backend development ,langchain,gemini api for chatbot.
        </p>

        <h2>Mission/Vision</h2>
        <p>
        My mission is to revolutionize the agriculture industry by providing farmers with advanced tools for disease prevention, early detection, and optimized crop management.
        </p>

        <h2>Contact Information</h2>
        <p class="contact-info">
        Email: abhisheksingh70209@gmail.com<br>
        Phone: +91 9540578309
        </p>

        <h2>Acknowledgments/Credits</h2>
        <p>
        We acknowledge the support and contributions of all th website like streamlit documentation,google gemini api,google ai studio,tensorflow .
        </p>

        <h2>Legal Information</h2>
        <p>
        Please review our terms of service and privacy policy for more information about data usage and user rights.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<h1 class="title-text">About Me</h1>', unsafe_allow_html=True)
    st.markdown("""
   # ABHISHEK SINGH
New Delhi, INDIA  
+91 9540578309  
email@abhisheksingh  
[LinkedIn](https://www.linkedin.com/in/abhishek-singh-5604b1243/) | [Portfolio](link_to_portfolio) | [Github](https://github.com/AbhishekSingh010?tab=repositories)

## EDUCATION  
**Maharaja Surajmal Institute**, New Delhi, IN  
Bachelor of Computer Applications (BCA) in Data Analytics  
GPA: 9.5/10  
July 2021 - June 2024  
- Machine Learning, Data Analytics, Business Analytics, Optimization Models, Recommender Systems, Statistics.

**Coursera**, Remote, IN  
IBM Data Science Professional Certification  
GPA: 9.9/10  
October 2023 - March 2024  
- Machine Learning, Data Analytics, SQL, Business Analytics, Data Science Methodology, Statistics.

## SKILLS & CERTIFICATIONS  
- **Programming:** Python, SQL (DBMS)  
- **Machine Learning Algorithms:** Multinomial regression, Decision Trees, SVM, KNN, Ensemble methods, Neural networks  
- **Cloud Services:** Amazon Web Services (AWS)  
- **Data Visualization:** Tableau, Power BI  
- **Tools and Frameworks:** Generative-AI, Google Analytics, Git, MS Excel, JIRA, MS 365, TensorFlow  
- **AI Solutions and Prompt Design Certifications:** IBM Data Science Professional certificate, Advance Excel  
- **Others:** Data Mining, Software Development Cycle, Django, Flask, Fast-API, Data Structure and Algorithm

"""

      f"[Download Resume]({resume_url})",

unsafe_allow_html=True
)

resume_url = "https://drive.google.com/file/d/1r1SaS6qPS60OQRolBgxrisnnuRHrrjS6/view?usp=sharing"
# button_clicked = st.button("Download Resume")
# if button_clicked:
#     st.markdown(f"[Download Resume]({resume_url})", unsafe_allow_html=True)
