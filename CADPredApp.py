import numpy as np
import pickle
import streamlit as st

# Load trained model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# Function for prediction
def CAD_prediction(input_data):
    inputAr = np.asarray(input_data, dtype=float)  # Convert to float
    inputAr_reshaped = inputAr.reshape(1, -1)  
    prediction = loaded_model.predict(inputAr_reshaped)

    if prediction[0] == 0:
        return "The Person does not have CAD"
    else:
        return "The Person has CAD"

# Web app
def main():
    st.title("ML Based CAD Prediction")

    # User inputs
    age = st.text_input("Age:")
    gender = st.text_input("Gender:")
    chestpain = st.text_input("Severity of Chest pain (0 for No pain, 1 to 3 as per severity):")
    restingBP = st.text_input("Resting Blood Pressure:")
    serumcholestrol = st.text_input("Serum Cholestrol Level:")
    fastingbloodsugar = st.text_input("Fasting Blood Sugar Level:")
    restingrelectro = st.text_input("Resting ECG Peak Value:")
    maxheartrate = st.text_input("Maximum Heart Rate:")
    exerciseangia = st.text_input("Angina during Exercising?:")
    oldpeak = st.text_input("Oldpeak:")
    slope = st.text_input("Slope of ECG:")
    noofmajorvessels = st.text_input("Number of major affected blood vessels:")

    # Convert inputs to numeric values
    if st.button("Click to Predict CAD"):
        try:
            input_data = [
                float(age), float(gender), float(chestpain), float(restingBP),
                float(serumcholestrol), float(fastingbloodsugar), float(restingrelectro),
                float(maxheartrate), float(exerciseangia), float(oldpeak),
                float(slope), float(noofmajorvessels)
            ]
            diagnosis = CAD_prediction(input_data)
            st.success(diagnosis)
        except ValueError:
            st.error("Please enter valid numeric inputs.")

if __name__ == '__main__':
    main()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
