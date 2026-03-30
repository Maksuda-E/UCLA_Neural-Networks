# This line imports streamlit for the web app
import streamlit as st

# This line imports the prediction function
from src.predict import predict_admission

# This line sets the page title and layout
st.set_page_config(page_title="Graduate Admission Prediction", layout="centered")

# This line shows the app title
st.title("Graduate Admission Prediction App")

# This line shows a short description
st.write("Enter applicant details to predict admission chance.")

# This line creates a numeric input for GRE score
gre_score = st.number_input("GRE Score", min_value=0.0, value=320.0)

# This line creates a numeric input for TOEFL score
toefl_score = st.number_input("TOEFL Score", min_value=0.0, value=110.0)

# This line creates a dropdown for university rating
university_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])

# This line creates a numeric input for SOP
sop = st.number_input("SOP Strength", min_value=0.0, max_value=5.0, value=4.0)

# This line creates a numeric input for LOR
lor = st.number_input("LOR Strength", min_value=0.0, max_value=5.0, value=4.0)

# This line creates a numeric input for CGPA
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.5)

# This line creates a dropdown for research experience
research = st.selectbox("Research Experience", [0, 1])

# This line checks if the Predict button is clicked
if st.button("Predict"):
    # This line creates the input dictionary
    user_input = {
        "GRE_Score": gre_score,
        "TOEFL_Score": toefl_score,
        "University_Rating": university_rating,
        "SOP": sop,
        "LOR": lor,
        "CGPA": cgpa,
        "Research": research
    }

    # This line starts a try block
    try:
        # This line gets the prediction result
        result = predict_admission(user_input)

        # This line displays the result
        st.success(f"Prediction Result: {result}")

    # This block handles app errors
    except Exception as exc:
        # This line shows the error message
        st.error(f"Prediction failed: {exc}")