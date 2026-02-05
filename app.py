import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

@st.cache_resource
def train_model():
    try:
        df = pd.read_csv("youtube_channel_real_performance_analytics.csv")
    except FileNotFoundError:
        return None, None

    
    columns_to_keep = [
        'Video Publish Time', 'Video Duration', 'Day of Week',
        'Impressions', 'Video Thumbnail CTR (%)', 'Views',
        'Average View Percentage (%)', 'Average View Duration', 'Watch Time (hours)',
        'Likes', 'Shares', 'New Comments', 'Like Rate (%)',
        'Subscribers', 'New Subscribers', 'Returning Viewers', 'New Viewers',
        'Estimated Revenue (USD)', 'Revenue per 1000 Views (USD)' 
    ]
    df_analysis = df[columns_to_keep].copy()
    day_mapping = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 
        'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    df_analysis['Day_Encoded'] = df_analysis['Day of Week'].map(day_mapping)

    X = df_analysis.drop(columns=['Video Publish Time', 'Day of Week', 'Views'])
    y = df_analysis['Views']
    model = RandomForestRegressor()
    model.fit(X, y)
    
    return model, X.columns

model, feature_names = train_model()

st.title(" YouTube Views Predictor")

if model is None:
    st.error("Error: CSV file not found. Please place 'youtube_channel_real_performance_analytics.csv' in the same folder.")
else:
    st.write("Enter the video stats below to predict how many **Views** it will get.")

    user_inputs = []
        for col in feature_names:
        if "Day_Encoded" in col:
            val = st.selectbox("Day of Week", options=[0,1,2,3,4,5,6], format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        else:
            val = st.number_input(f"Enter {col}", value=0.0)
        user_inputs.append(val)

    if st.button("Predict Views"):
        final_input = np.array([user_inputs])
        
        prediction = model.predict(final_input)
        
        # Show result

        st.success(f"📈 Predicted Views: {int(prediction[0]):,}")
