import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load('model_diabetes.pkl')
scaler = joblib.load('scaler.pkl')

# Sidebar Navigasi
halaman = st.sidebar.radio("Pilih Halaman", ["📊 Dataset", "📈 Hasil Model", "📝 Prediksi"])

# Halaman Dataset
if halaman == "📊 Dataset":
    st.title("📊 Dataset Diabetes")
    data = pd.read_csv("diabetes.csv")
    st.write("Contoh Data:")
    st.dataframe(data.head())
    st.bar_chart(data['Age'].value_counts().sort_index())

# Halaman Hasil Model
elif halaman == "📈 Hasil Model":
    st.title("📈 Hasil Model")
    st.markdown("- Model: **Random Forest**")
    st.markdown("- Akurasi: **78%**")
    st.success("Model berhasil dimuat!")

# Halaman Prediksi
elif halaman == "📝 Prediksi":
    st.title("📝 Prediksi Diabetes")
    
    # Input User
    Pregnancies = st.number_input("Kehamilan", 0, 20, 1)
    Glucose = st.number_input("Glukosa", 0, 200, 100)
    BloodPressure = st.number_input("Tekanan Darah", 0, 122, 70)
    SkinThickness = st.number_input("Ketebalan Kulit", 0, 100, 20)
    Insulin = st.number_input("Insulin", 0, 900, 80)
    BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
    DPF = st.number_input("Riwayat Keluarga (DPF)", 0.0, 3.0, 0.5)
    Age = st.number_input("Umur", 0, 100, 30)

    data_input = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DPF, Age]])
    data_scaled = scaler.transform(data_input)

    if st.button("🔍 Prediksi"):
        prediksi = model.predict(data_scaled)
        if prediksi[0] == 1:
            st.error("⚠️ Pasien berisiko diabetes.")
        else:
            st.success("✅ Pasien tidak berisiko diabetes.")
