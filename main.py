# Import streamlit for building the web application
import streamlit as st

# Import the prediction function
from src.predict import predict_admission

# Import required libraries for loading saved metrics
import json
import os

# Configure the page settings such as title, icon, and layout
st.set_page_config(
    page_title="Graduate Admission Prediction",
    layout="wide"
)

# Apply custom CSS styling for better UI appearance
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #eef2ff, #f8fafc, #ecfeff);
    }

    .title {
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        color: #0f172a;
    }

    .subtitle {
        text-align: center;
        color: #475569;
        margin-bottom: 25px;
    }

    div.stButton > button {
        width: 100%;
        height: 45px;
        border-radius: 10px;
        border: none;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
        font-size: 16px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #6d28d9);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the main title of the application
st.markdown('<div class="title">Graduate Admission Prediction</div>', unsafe_allow_html=True)

# Display subtitle text below the title
st.markdown(
    '<div class="subtitle">Enter applicant details to estimate admission probability</div>',
    unsafe_allow_html=True
)

# Create two columns where left is wider for inputs and right is for summary
left_col, right_col = st.columns([2, 1])

# Left column contains all user input fields and prediction result
with left_col:

    # Create two inner columns for organizing input fields
    col1, col2 = st.columns(2)

    # First column inputs
    with col1:
        # Input for GRE score
        gre_score = st.number_input(
            "GRE Score",
            min_value=260,
            max_value=340,
            value=320
        )

        # Dropdown for university rating
        university_rating = st.selectbox(
            "University Rating",
            [1, 2, 3, 4, 5]
        )

        # Slider for LOR strength
        lor = st.slider(
            "LOR Strength",
            min_value=1.0,
            max_value=5.0,
            value=4.0
        )

        # Dropdown for research experience
        research = st.selectbox(
            "Research Experience",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    # Second column inputs
    with col2:
        # Input for TOEFL score
        toefl_score = st.number_input(
            "TOEFL Score",
            min_value=0,
            max_value=120,
            value=110
        )

        # Slider for SOP strength
        sop = st.slider(
            "SOP Strength",
            min_value=1.0,
            max_value=5.0,
            value=4.0
        )

        # Input for CGPA
        cgpa = st.number_input(
            "CGPA",
            min_value=0.0,
            max_value=10.0,
            value=8.5
        )

    # Add spacing before button
    st.markdown("<br>", unsafe_allow_html=True)

    # Create a button for prediction
    predict_button = st.button("Predict Admission Chance")

    # Check if the button is clicked
    if predict_button:

        # Prepare input data dictionary for model
        user_input = {
            "GRE_Score": gre_score,
            "TOEFL_Score": toefl_score,
            "University_Rating": university_rating,
            "SOP": sop,
            "LOR": lor,
            "CGPA": cgpa,
            "Research": research
        }

        # Try to get prediction from model
        try:
            result = predict_admission(user_input)

            # Try converting result to numeric for percentage display
            try:
                value = float(result)

                # Convert probability to percentage if needed
                if value <= 1:
                    value = value * 100

                # Display result as success message
                st.success(f"Admission Chance: {value:.2f}%")

                # Show progress bar based on prediction value
                st.progress(int(value))

            # If result is not numeric, display raw result
            except:
                st.success(f"Result: {result}")

        # Handle prediction errors
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Right column displays model summary and metrics
with right_col:

    # Display summary title
    st.markdown("### Model Summary")

    # Try loading saved model metrics
    try:
        metrics_path = "artifacts/metrics.json"

        # Check if metrics file exists
        if os.path.exists(metrics_path):

            # Load metrics from JSON file
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            # Display metrics heading
            st.markdown("#### Evaluation Metrics")

            # Loop through metrics and display each one
            for key, value in metrics.items():
                st.metric(label=key, value=round(value, 4))

        # If file not found, show message
        else:
            st.info("Run training pipeline first to generate metrics")

    # Handle errors in loading metrics
    except Exception as e:
        st.error(f"Error loading summary: {e}")

    # Display additional model information
    st.markdown("#### Model Info")
    st.write("Model: Linear Regression")
    st.write("Scaler: StandardScaler")
