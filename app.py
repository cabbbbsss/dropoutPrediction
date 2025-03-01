import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load model dan fitur yang digunakan
with open("model.pkl", "rb") as file:
    model_data = pickle.load(file)
    model = model_data["model"]  # Ambil objek model yang benar
    selected_features = model_data["features"]

# Load dataset untuk referensi preprocessing
df = pd.read_csv("data.csv", sep=';')

# List fitur kategorikal dan numerik
categorical_columns = ["Marital_status", "Application_mode", "Course", "Daytime_evening_attendance", "Previous_qualification", "Nacionality", "Mothers_qualification", "Fathers_qualification", "Mothers_occupation", "Fathers_occupation", "Displaced", "Educational_special_needs", "Debtor", "Tuition_fees_up_to_date", "Gender", "Scholarship_holder", "International"]
numerical_columns = ["Previous_qualification_grade", "Admission_grade", "Age_at_enrollment", "Curricular_units_1st_sem_credited", "Curricular_units_1st_sem_enrolled", "Curricular_units_1st_sem_evaluations", "Curricular_units_1st_sem_approved", "Curricular_units_1st_sem_grade", "Curricular_units_2nd_sem_credited", "Curricular_units_2nd_sem_enrolled", "Curricular_units_2nd_sem_evaluations", "Curricular_units_2nd_sem_approved", "Curricular_units_2nd_sem_grade", "Unemployment_rate", "Inflation_rate", "GDP"]

# Preprocessing function
def preprocess_input(input_data):
    # Encode categorical features
    encoder = LabelEncoder()
    for col in categorical_columns:
        input_data[col] = encoder.fit_transform([input_data[col]])[0]
    
    # Scale numerical features
    scaler = StandardScaler()
    for col in numerical_columns:
        input_data[col] = scaler.fit_transform([[input_data[col]]])[0][0]
    
    return pd.DataFrame([input_data])

# Streamlit UI
st.title("Prediksi Dropout Mahasiswa")

# Input fitur oleh pengguna
st.sidebar.header("Masukkan Data Mahasiswa")
input_data = {}
for col in categorical_columns:
    input_data[col] = st.sidebar.selectbox(f"{col}", df[col].unique())
for col in numerical_columns:
    input_data[col] = st.sidebar.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

# Konversi input ke dataframe
input_df = preprocess_input(input_data)

# Menyesuaikan fitur input agar sesuai dengan model
input_df = input_df.reindex(columns=selected_features, fill_value=0)

# Prediksi Dropout
if st.sidebar.button("Prediksi"):
    pred_proba = model.predict_proba(input_df)[0, 1]  # Probabilitas dropout
    st.write(f"Probabilitas Dropout: {pred_proba:.2f}")
    if pred_proba > 0.5:
        st.error("Mahasiswa berisiko tinggi untuk dropout!")
    else:
        st.success("Mahasiswa tidak berisiko tinggi untuk dropout.")