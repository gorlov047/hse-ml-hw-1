import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title("Car Price Prediction")

@st.cache_resource
def load_model():
    with open('car_price_model.pkl', 'rb') as f:
        return pickle.load(f)

def extract_torque(torque_str):
    if pd.isna(torque_str):
        return np.nan, np.nan
    
    torque_str = str(torque_str).strip().lower()
    torque_value, rpm_value = np.nan, np.nan
    
    kgm_match = re.search(r'(\d+\.?\d*)\s*(?:kg\.?m|kgm|kg)', torque_str)
    if kgm_match:
        torque_kgm = float(kgm_match.group(1))
        torque_value = torque_kgm * 9.80665
    
    nm_match = re.search(r'(\d+\.?\d*)\s*(?:nm|n\.m)', torque_str)
    if nm_match and pd.isna(torque_value):
        torque_value = float(nm_match.group(1))
    
    range_match = re.search(r'@?\s*(\d+)[\s\-–]+(\d+)\s*(?:rpm)?', torque_str)
    if range_match:
        rpm1 = float(range_match.group(1))
        rpm2 = float(range_match.group(2))
        rpm_value = (rpm1 + rpm2) / 2
    
    elif re.search(r'@', torque_str):
        single_match = re.search(r'@\s*(\d+\.?\d*)\s*(?:rpm)?', torque_str)
        if single_match:
            rpm_value = float(single_match.group(1))

    return torque_value, rpm_value

def preprocess_input(df, artifacts):
    df = df.copy()
    
    
    if 'mileage' in df.columns and df['mileage'].dtype == 'object':
        df['mileage'] = df['mileage'].str.extract(r'(\d+\.?\d*)').astype(float)
    
    if 'engine' in df.columns and df['engine'].dtype == 'object':
        df['engine'] = df['engine'].str.extract(r'(\d+)').astype(float)
    
    if 'max_power' in df.columns and df['max_power'].dtype == 'object':
        df['max_power'] = df['max_power'].str.extract(r'(\d+\.?\d*)').astype(float)
    
    if 'torque' in df.columns and df['torque'].dtype == 'object':
        df[["torque", "max_torque_rpm"]] = df["torque"].apply(extract_torque).apply(pd.Series)
    
    
    for col, median_val in artifacts['medians'].items():
        if col in df.columns:
            df[col] = df[col].fillna(median_val)
    
    
    if 'engine' in df.columns:
        df['engine'] = df['engine'].astype(int)
    
    
    if 'seats' in df.columns:
        df['seats'] = df['seats'].astype(str)
        rare_seats = artifacts['rare_categories'].get('seats', [])
        df['seats'] = df['seats'].apply(lambda x: x if x not in rare_seats else 'Other')
    
    
    if 'name' in df.columns:
        df['name'] = df['name'].apply(lambda x: str(x).split()[0] if pd.notnull(x) else 'Unknown')
        rare_names = artifacts['rare_categories'].get('name', [])
        df['name'] = df['name'].apply(lambda x: x if x not in rare_names else 'Other')
    
    return df

def predict_single(input_data, artifacts):
    
    input_processed = preprocess_input(input_data, artifacts)
    
    
    scaler = artifacts['scaler']
    ohe = artifacts['ohe']
    model = artifacts['model']
    numeric_features = artifacts['numeric_features']
    categorical_features = artifacts['categorical_features']
    
    try:
        
        X_num = scaler.transform(input_processed[numeric_features])
        
        
        X_cat = ohe.transform(input_processed[categorical_features]).toarray()
        
        
        X_combined = np.concatenate([X_num, X_cat], axis=1)
        
        
        prediction = model.predict(X_combined)
        return prediction[0]
    except Exception as e:
        st.error(f"Ошибка предсказания: {str(e)}")
        return None


try:
    artifacts = load_model()
except:
    st.error("Не удалось загрузить модель. Убедитесь что файл car_price_model.pkl существует")
    st.stop()


tab1, tab2 = st.tabs(["Ручной ввод", "Загрузка CSV"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Марка", "Maruti")
        year = st.number_input("Год", min_value=1990, max_value=2023, value=2015)
        km_driven = st.number_input("Пробег (км)", min_value=0, value=50000)
        fuel = st.selectbox("Топливо", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
        seller_type = st.selectbox("Продавец", ["Individual", "Dealer", "Trustmark Dealer"])
    
    with col2:
        transmission = st.selectbox("Коробка", ["Manual", "Automatic"])
        owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
        mileage = st.number_input("Расход (kmpl)", min_value=0.0, value=20.0)
        engine = st.number_input("Объем двигателя (CC)", min_value=0, value=1200)
        max_power = st.number_input("Мощность (bhp)", min_value=0.0, value=80.0)
    
    col3, col4 = st.columns(2)
    
    with col3:
        seats = st.selectbox("Места", [2, 4, 5, 6, 7, 8, 9, 10], index=3)
        torque_input = st.text_input("Крутящий момент", "120Nm@4000rpm")
    
    if st.button("Предсказать цену"):
        input_data = pd.DataFrame({
            'name': [name],
            'year': [year],
            'km_driven': [km_driven],
            'fuel': [fuel],
            'seller_type': [seller_type],
            'transmission': [transmission],
            'owner': [owner],
            'mileage': [mileage],
            'engine': [engine],
            'max_power': [max_power],
            'seats': [seats],
            'torque': [torque_input]
        })
        
        prediction = predict_single(input_data, artifacts)
        if prediction is not None:
            st.success(f"Предсказанная цена: ₹{prediction:,.2f}")

with tab2:
    uploaded_file = st.file_uploader("Загрузите CSV файл", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Предпросмотр данных:")
        st.dataframe(df.head())
        
        required_cols = artifacts['numeric_features'] + artifacts['categorical_features']
        required_cols = list(set(required_cols))
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Отсутствуют колонки: {missing_cols}")
        else:
            if st.button("Предсказать цены"):
                predictions = []
                
                for i, row in df.iterrows():
                    input_data = pd.DataFrame([row])
                    pred = predict_single(input_data, artifacts)
                    predictions.append(pred if pred is not None else np.nan)
                
                df['predicted_price'] = predictions
                
                st.write("Результаты:")
                st.dataframe(df[['name', 'year', 'predicted_price']].head())
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Скачать предсказания",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )