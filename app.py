import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Housing Price Predictor", page_icon="🏠", layout="wide")

st.title("🏠 Housing Price Prediction")
st.markdown("**Machine Learning Mini Model by Sandra Mkanyi**")
st.markdown("---")

@st.cache_resource
def train_model():
    data = {
        'size_sqft': [1500, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3500, 1200, 1600, 1900, 2100, 2300, 2500, 2700, 2900, 3100, 3300],
        'bedrooms': [3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5],
        'bathrooms': [2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4],
        'age_years': [10, 5, 15, 8, 3, 12, 6, 2, 20, 4, 25, 8, 12, 7, 2, 18, 9, 4, 11, 6],
        'price_usd': [300000, 350000, 400000, 420000, 450000, 470000, 520000, 550000, 480000, 580000, 250000, 320000, 380000, 410000, 440000, 460000, 500000, 530000, 490000, 560000]
    }
    df = pd.DataFrame(data)
    X = df[['size_sqft', 'bedrooms', 'bathrooms', 'age_years']]
    y = df['price_usd']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)
    return model, scaler, df

model, scaler, df = train_model()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("📝 Input Features")
    size_sqft = st.slider("🏡 House Size (sqft)", 1000, 4000, 2000, 100)
    bedrooms = st.slider("🛏️ Bedrooms", 1, 6, 3)
    bathrooms = st.slider("🚿 Bathrooms", 1, 5, 2)
    age_years = st.slider("📅 Age (years)", 0, 30, 5)
    predict_button = st.button("🔮 Predict Price", type="primary")

with col2:
    st.header("📊 Model Information")
    st.info("Linear Regression model for housing price prediction")
    
    if predict_button:
        input_data = np.array([[size_sqft, bedrooms, bathrooms, age_years]])
        input_scaled = scaler.transform(input_data)
        predicted_price = model.predict(input_scaled)[0]
        
        st.success("### 🎯 Prediction Complete!")
        st.markdown(f"## **${predicted_price:,.2f}**")
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Size", f"{size_sqft:,} sqft")
            st.metric("Bedrooms", bedrooms)
        with c2:
            st.metric("Bathrooms", bathrooms)
            st.metric("Age", f"{age_years} yrs")
    else:
        st.warning("👆 Click 'Predict Price' to see results!")
        st.dataframe(df.head(5))

st.markdown("---")
st.caption("Built with ❤️ by Sandra Mkanyi")