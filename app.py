# Import streamlit for building the web app
import streamlit as st

# Import prediction function
from src.predict import predict_admission

# Set page configuration (title, icon, layout)
st.set_page_config(
    page_title="Graduate Admission Predictor",
    page_icon="🎓",
    layout="centered"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #eef2ff, #f8fafc, #ecfeff);
    }

    .title {
        font-size: 2.4rem;
        font-weight: 800;
        text-align: center;
        color: #0f172a;
    }

    .subtitle {
        text-align: center;
        color: #475569;
        margin-bottom: 25px;
    }

    .card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    }

    .result {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        padding: 15px;
        border-radius: 15px;
        color: white;
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
        margin-top: 15px;
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

# Display title
st.markdown('<div class="title">Graduate Admission Prediction</div>', unsafe_allow_html=True)

# Display subtitle
st.markdown('<div class="subtitle">Enter applicant details to estimate admission probability</div>', unsafe_allow_html=True)

# Create a styled container (card)
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    # Left column inputs
    with col1:
        gre_score = st.number_input(
            "GRE Score",
            min_value=260,
            max_value=340,
            value=320
        )

        university_rating = st.selectbox(
            "University Rating",
            [1, 2, 3, 4, 5]
        )

        lor = st.slider(
            "LOR Strength",
            min_value=1.0,
            max_value=5.0,
            value=4.0
        )

        research = st.selectbox(
            "Research Experience",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    # Right column inputs
    with col2:
        toefl_score = st.number_input(
            "TOEFL Score",
            min_value=0,
            max_value=120,
            value=110
        )

        sop = st.slider(
            "SOP Strength",
            min_value=1.0,
            max_value=5.0,
            value=4.0
        )

        cgpa = st.number_input(
            "CGPA",
            min_value=0.0,
            max_value=10.0,
            value=8.5
        )

    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)

    # Predict button
    predict_button = st.button("Predict Admission Chance")

    # When button is clicked
    if predict_button:

        # Prepare input dictionary for model
        user_input = {
            "GRE_Score": gre_score,
            "TOEFL_Score": toefl_score,
            "University_Rating": university_rating,
            "SOP": sop,
            "LOR": lor,
            "CGPA": cgpa,
            "Research": research
        }

        try:
            # Get prediction from model
            result = predict_admission(user_input)

            # Convert result into percentage if needed
            try:
                value = float(result)

                # If model returns probability (0–1), convert to %
                if value <= 1:
                    value = value * 100

                # Show styled result
                st.markdown(
                    f'<div class="result">Admission Chance: {value:.2f}%</div>',
                    unsafe_allow_html=True
                )

                # Show progress bar
                st.progress(int(value))

            except:
                # If result is not numeric, show raw output
                st.markdown(
                    f'<div class="result">Result: {result}</div>',
                    unsafe_allow_html=True
                )

        # Handle errors
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Close card container
    st.markdown('</div>', unsafe_allow_html=True)
