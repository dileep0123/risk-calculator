import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained models (dictionary of models)
models = joblib.load("all_target_models.pkl")  # Dictionary containing models for each target
smote_data = pd.read_csv("smote_data.csv")  # Dataset for preprocessing

# Display Main Title
st.markdown("<h1 style='text-align: center; color: yellow;'>RISK CALCULATOR</h1>", unsafe_allow_html=True)

# Add background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url(https://medicinetoday.com.au/sites/default/files/styles/wide/public/2024-06/0624CKD_CH.jpg.webp?itok=0n-LUOyv);
        background-size: cover;  
        background-position: center;  
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Subtitle for User Information
st.markdown("<h2 style='text-align: left; color: white;'>USER INFORMATION</h2>", unsafe_allow_html=True)

#Age
st.markdown("<h5><b>Enter Your Age:</b></h5>", unsafe_allow_html=True)
Age = st.number_input("a", min_value=0, max_value=100, step=1, value=0,label_visibility="collapsed")

# Gender
gender_options = ["Select Gender", "Male 🧑‍🦱", "Female 👩‍🦰"]
gender_mapping = {"Male 🧑‍🦱": 1, "Female 👩‍🦰": 0}
st.markdown("<h5><b>Select Your Gender 👨‍🦱👩‍🦰:</b></h5>", unsafe_allow_html=True)
Gender = st.selectbox("g", gender_options,label_visibility="collapsed")
gender_value = gender_mapping.get(Gender, None) if Gender != "Select Gender" else None

# Smoking
smoking_options = ["Non-smoker 🚭", "Occasional smoker 🚬", "Regular smoker 💨"]
smoking_mapping = {"Non-smoker 🚭": 0, "Occasional smoker 🚬": 1, "Regular smoker 💨": 2}
st.markdown("<h5><b>Select Your Smoking Habit:</b></h5>", unsafe_allow_html=True)
Smoking = st.radio("a", smoking_options,label_visibility="collapsed")
smoking_value = smoking_mapping.get(Smoking, None)

# Alcohol Consumption

st.markdown("<h5><b>Alcohol Consumption (Level):</b></h5>", unsafe_allow_html=True,)
alcohol_consumption = st.slider("a", min_value=0, max_value=20, value=0,label_visibility="collapsed")

# Cholesterol
st.markdown("<h5><b>Enter Your Cholesterol Level (mg/dl):</b></h5>", unsafe_allow_html=True,)
cholesterol = st.number_input("c", min_value=0, max_value=600, step=1, value=0,label_visibility="collapsed")

# Chest Pain
chest_pain_options = ["No", "Stage 1", "Stage 2", "Stage 3"]
chest_pain_mapping = {"No": 0, "Stage 1": 1, "Stage 2": 2, "Stage 3": 3}
st.markdown("<h5><b>Select Your Chest Pain Type::</b></h5>", unsafe_allow_html=True)
chest_pain = st.radio("c", chest_pain_options,label_visibility="collapsed")
chest_pain_value = chest_pain_mapping.get(chest_pain, None)

# Trestbps
st.markdown("<h5><b>Enter Your Resting Blood Pressure (Trestbps):</b></h5>", unsafe_allow_html=True)
trestbps_value = st.number_input("t", min_value=80, max_value=300, step=1, value=120,label_visibility="collapsed")

# Fbs
fbs_options = ["No", "Yes"]
fbs_mapping = {"No": 0, "Yes": 1}
st.markdown("<h5><b>Is Your Fasting Blood Sugar Greater Than 120 mg/dl?</b></h5>", unsafe_allow_html=True)
Fbs = st.selectbox("fbs", fbs_options,label_visibility="collapsed")
fbs_value = fbs_mapping.get(Fbs, None)

# Max Heart Rate
st.markdown("<h5><b>Enter Your Max Heart Rate::</b></h5>", unsafe_allow_html=True)
max_heart_rate = st.slider("hr", min_value=70, max_value=250, value=120,label_visibility="collapsed")

# Physical Activity
physical_activity_options = [
    "Very Low", "Low", "Moderate", "Active", "Very Active",
    "Extremely Active", "Sedentary", "Occasionally Active", "Highly Active"
]
physical_activity_mapping = {
    "Very Low": 0, "Low": 1, "Moderate": 2, "Active": 3, "Very Active": 4,
    "Extremely Active": 5, "Sedentary": 6, "Occasionally Active": 7, "Highly Active": 8

}
st.markdown("<h5><b>Select Your Physical Activity Level:</b></h5>", unsafe_allow_html=True)
Physical_Activity = st.selectbox("ppp", physical_activity_options,label_visibility="collapsed")
physical_activity_value = physical_activity_mapping.get(Physical_Activity, None)

# Predict Button
if st.button("Predict Risk"):
    # Prepare Data for Prediction
    input_data = pd.DataFrame({
        "Age": [Age],
        "Gender": [gender_value],
        "Smoking": [smoking_value],
        "Alcohol Consumption": [alcohol_consumption],
        "Cholesterol": [cholesterol],
        "Chest Pain": [chest_pain_value],
        "Trestbps": [trestbps_value],
        "Fbs": [fbs_value],
        "Max Heart Rate": [max_heart_rate],
        "Physical Activity": [physical_activity_value]
    })

    # Debugging: Display Input Data (Optional)
    st.write("Input Data for Prediction:")
    st.write(input_data)

    # Check if all fields are filled
    if input_data.isnull().values.any():
        st.warning("Please fill all the fields to calculate your risk.")
    else:
        # Predict Risks for Heart, Kidney, and Lung Diseases
        heart_model = models['Heart Disease']  
        kidney_model = models['Kidney Disease'] 
        lung_model = models['Lung Cancer']  

        heart_risk = heart_model.predict_proba(input_data)[0][1] * 100
        kidney_risk = kidney_model.predict_proba(input_data)[0][1] * 100
        lung_risk = lung_model.predict_proba(input_data)[0][1] * 100


        heart_threshold = 30  
        kidney_threshold = 25  
        lung_threshold = 20  

        def get_recommendations(disease, risk_value, threshold):
            if risk_value < threshold:
                risk_level = "Low Risk"
                if disease == "Heart Disease":
                    recommendations = (
                        "For Heart Disease: Maintain a healthy diet (fruits, vegetables, lean proteins) and active lifestyle. "
                        "Regular checkups are recommended. Engage in 150 minutes of moderate exercise weekly, like walking or cycling."
                    )
                elif disease == "Kidney Disease":
                    recommendations = (
                        "For Kidney Disease: Follow a balanced diet, stay hydrated, and engage in moderate exercise (walking, swimming). "
                        "Monitor kidney health periodically and avoid smoking and excessive alcohol."
                    )
                elif disease == "Lung Cancer":
                    recommendations = (
                        "For Lung Cancer: Maintain a diet rich in antioxidants (fruits, vegetables), avoid smoking, and stay active with regular exercise. "
                        "Consider lung health screenings based on family history and avoid exposure to air pollution."
                    )
            elif threshold <= risk_value < 60:
                risk_level = "Medium Risk"
                if disease == "Heart Disease":
                    recommendations = (
                        "For Heart Disease: Follow a balanced diet with omega-3s and fiber, engage in moderate physical activity, "
                        "and have routine medical checkups. Monitor blood pressure, cholesterol, and blood sugar."
                    )
                elif disease == "Kidney Disease":
                    recommendations = (
                        "For Kidney Disease: Limit high-potassium foods and monitor kidney function. Incorporate more plant-based foods and limit protein intake. "
                        "Stay active with moderate exercise and maintain a healthy weight."
                    )
                elif disease == "Lung Cancer":
                    recommendations = (
                        "For Lung Cancer: Increase intake of cruciferous vegetables and anti-inflammatory foods, engage in regular cardio exercises, "
                        "and consult a healthcare provider for screenings. Avoid smoking and environmental toxins."
                    )
            else:
                risk_level = "High Risk"
                if disease == "Heart Disease":
                    recommendations = (
                        "For Heart Disease: include plenty of fruits, vegetables, healthy fats, chicken and fish, engage in regular physical activity, and consult your doctor immediately. "
                        "Monitor blood pressure, cholesterol, and blood sugar levels frequently."
                    )
                elif disease == "Kidney Disease":
                    recommendations = (
                        "For Kidney Disease: Follow a renal-friendly diet (controlled protein, sodium, potassium). Avoid processed foods and stay active with light exercises. "
                        "Frequent kidney function tests and medications as prescribed."
                    )
                elif disease == "Lung Cancer":
                    recommendations = (
                        "For Lung Cancer: Follow a diet rich in vitamins A, C, and E, stay active with light exercises, and avoid smoking. "
                        "Consult your doctor immediately for screenings if necessary."
                    )

            return risk_level, recommendations

        def simulate_risk_decrease(risk_value, months):
            """Simulate risk decrease over a given period of months"""
            risk_decrease_rate = 0.05  # 5% reduction per month for high risk
            for month in range(months):
                if risk_value > 0:
                    risk_value -= risk_value * risk_decrease_rate
            return risk_value  

        # Classify Risks and Get Recommendations
        heart_risk_level, heart_recommendation = get_recommendations("Heart Disease", heart_risk, 30)
        kidney_risk_level, kidney_recommendation = get_recommendations("Kidney Disease", kidney_risk, 25)
        lung_risk_level, lung_recommendation = get_recommendations("Lung Cancer", lung_risk, 20)

        # Display Results
        st.markdown("<h2 style='color: white;'>Disease Risk Predictions:</h2>", unsafe_allow_html=True)
        st.write(f"**Heart Disease Risk:** {heart_risk:.2f}% - {heart_risk_level}")
        st.write(f"**Kidney Disease Risk:** {kidney_risk:.2f}% - {kidney_risk_level}")
        st.write(f"**Lung Cancer Risk:** {lung_risk:.2f}% - {lung_risk_level}")

        # Display Recommendations
        st.markdown("<h3 style='color: white;'>Health Recommendations:</h3>", unsafe_allow_html=True)
        st.write(f" {heart_recommendation}")
        st.write(f" {kidney_recommendation}")
        st.write(f"{lung_recommendation}")

        st.markdown("<h3 style='color: white;'>Risk Reduction After One year:</h3>", unsafe_allow_html=True)
        
        months_input = 12

    
        # Calculate and display the updated risks after the specified number of months
        updated_heart_risk = simulate_risk_decrease(heart_risk, months_input)
        updated_kidney_risk = simulate_risk_decrease(kidney_risk, months_input)
        updated_lung_risk = simulate_risk_decrease(lung_risk, months_input)

        st.write(f" Heart Disease Risk after {months_input} months: {updated_heart_risk:.2f}%")
        st.write(f" Kidney Disease Risk after {months_input} months: {updated_kidney_risk:.2f}%")
        st.write(f" Lung Cancer Risk after {months_input} months: {updated_lung_risk:.2f}%")

        # Plotting the risk percentage before and after health recommendations
        fig, ax = plt.subplots(figsize=(5, 5))
        diseases = ["Heart Disease", "Kidney Disease", "Lung Cancer"]
        initial_risks = [heart_risk, kidney_risk, lung_risk]
        updated_risks = [updated_heart_risk, updated_kidney_risk, updated_lung_risk]

        ax.bar(diseases, initial_risks, color='blue', alpha=0.7, label='Initial Risk')
        ax.bar(diseases, updated_risks, color='red', alpha=0.7, label='Updated Risk')

        ax.set_xlabel("Disease")
        ax.set_ylabel("Risk (%)")
        ax.set_title("Risk Reduction Over Time (Before vs After Health Recommendations)")
        ax.legend()

        st.pyplot(fig)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        months = list(range(1, 13))
        ax.plot(months, [simulate_risk_decrease(heart_risk, month) for month in months], label='Heart Disease', color='blue')
        ax.plot(months, [simulate_risk_decrease(kidney_risk, month) for month in months], label='Kidney Disease', color='green')
        ax.plot(months, [simulate_risk_decrease(lung_risk, month) for month in months], label='Lung Cancer', color='red')

        ax.set_xlabel('Months')
        ax.set_ylabel('Risk (%)')
        ax.set_title('Risk Reduction Over 12 Months')
        ax.legend()

        st.pyplot(fig)
        