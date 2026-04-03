#  imports streamlit so the web application interface can be built.
import streamlit as st

#  imports the prediction function from the project prediction module.
from src.predict import predict_admission

#  sets the browser tab title and uses a wide layout for a cleaner interface.
st.set_page_config(page_title="UCLA Admission Predictor", layout="wide")

#  stores custom CSS styling for the page, cards, and result boxes.
custom_css = """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f3c88;
        margin-bottom: 0.25rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #4a4a4a;
        margin-bottom: 1.5rem;
    }
    .metric-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1f3c88;
    }
</style>
"""

#  injects the custom CSS into the Streamlit page.
st.markdown(custom_css, unsafe_allow_html=True)

#  displays the main project title.
st.markdown('<div class="main-title">UCLA Graduate Admission Predictor</div>', unsafe_allow_html=True)

#  displays a short description below the main title.
st.markdown(
    '<div class="sub-title">This app predicts whether an applicant has a high or low chance of admission based on the trained neural network model.</div>',
    unsafe_allow_html=True
)

#  creates two main columns so inputs and results can be separated neatly.
left_col, right_col = st.columns([2, 1])

# This block starts the left column where all user inputs will be placed.
with left_col:

    #  shows a heading for the applicant profile section.
    st.subheader("Applicant Profile")

    #  creates two sub-columns so the input fields look balanced.
    input_col_1, input_col_2 = st.columns(2)

    # This block contains the first half of the input controls.
    with input_col_1:

        #  creates a numeric input for GRE score.
        gre_score = st.number_input("GRE Score", min_value=260, max_value=340, value=320, step=1)

        #  creates a dropdown for university rating.
        university_rating = st.selectbox("University Rating", options=[1, 2, 3, 4, 5], index=2)

        #  creates a slider for statement of purpose score.
        sop = st.slider("SOP Strength", min_value=1.0, max_value=5.0, value=3.5, step=0.5)

        #  creates a slider for letter of recommendation score.
        lor = st.slider("LOR Strength", min_value=1.0, max_value=5.0, value=3.5, step=0.5)

    # This block contains the second half of the input controls.
    with input_col_2:

        #  creates a numeric input for TOEFL score.
        toefl_score = st.number_input("TOEFL Score", min_value=80, max_value=120, value=107, step=1)

        #  creates a numeric input for CGPA.
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.5, step=0.01, format="%.2f")

        #  creates a dropdown for research experience.
        research_label = st.selectbox("Research Experience", options=["No", "Yes"], index=1)

        #  converts the displayed research label into the encoded numeric value used by the model.
        research = 1 if research_label == "Yes" else 0

    #  creates a centered prediction button.
    predict_clicked = st.button("Predict Admission Category", use_container_width=True)

    #  closes the styled container for the input section.
    st.markdown('</div>', unsafe_allow_html=True)

# This block starts the right column where explanation and output will be shown.
with right_col:

    #  displays a heading for model information.
    st.subheader("Model Information")

    #  explains the target transformation used in the notebook.
    st.write("Target rule: Admit_Chance is converted to 1 when the value is at least 0.80, otherwise 0.")

    #  explains the model type used in training.
    st.write("Model: MLPClassifier")

    #  explains the scaler used in training.
    st.write("Scaler: MinMaxScaler")

    #  explains the categorical encoding used in preprocessing.
    st.write("Categorical handling: One hot encoding for University_Rating and Research")

    #  closes the styled container for project information.
    st.markdown('</div>', unsafe_allow_html=True)

# This block runs only when the user clicks the prediction button.
if predict_clicked:

    #  creates a dictionary containing the raw user inputs.
    user_input = {
        "GRE_Score": gre_score,
        "TOEFL_Score": toefl_score,
        "University_Rating": university_rating,
        "SOP": sop,
        "LOR": lor,
        "CGPA": cgpa,
        "Research": research
    }

    #  starts a try block so errors can be shown cleanly in the app.
    try:

        #  sends the input dictionary to the prediction pipeline and stores the response.
        prediction_result = predict_admission(user_input)

        #  extracts the predicted class label from the prediction response.
        predicted_label = prediction_result["predicted_label"]

        #  extracts the probability of the positive class from the prediction response.
        positive_probability = prediction_result["probability_of_high_admission"]

        #  converts the probability into percentage format for display.
        probability_percent = positive_probability * 100

        #  shows a heading for the prediction result section.
        st.subheader("Prediction Result")

        #  displays the predicted admission class.
        st.write(f"Predicted Class: {predicted_label}")

        #  displays the probability of high admission as a formatted percentage.
        st.write(f"Probability of High Admission: {probability_percent:.2f}%")

        #  draws a progress bar using the calculated percentage.
        st.progress(int(probability_percent))

        #  shows a success message if the model predicts a high chance of admission.
        if predicted_label == "High Chance of Admission":
            st.success("The model predicts that this student has a high chance of admission.")
        else:
            st.warning("The model predicts that this student has a low chance of admission.")

        #  closes the styled result container.
        st.markdown('</div>', unsafe_allow_html=True)

    # This block catches any exception raised during prediction.
    except Exception as error:

        #  shows an error message in the Streamlit interface.
        st.error(f"Prediction failed: {error}")