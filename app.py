import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# --- 1. Load Data & Train Model ---
# We use @st.cache_resource so the model trains only once, making the app fast.
@st.cache_resource
def train_model():
    # Load dataset
    try:
        df = pd.read_csv("youtube_channel_real_performance_analytics.csv")
    except FileNotFoundError:
        return None, None

    # --- Data Cleaning (Matches your Notebook) ---
    # We keep only the columns you used in training
    columns_to_keep = [
        'Video Publish Time', 'Video Duration', 'Day of Week',
        'Impressions', 'Video Thumbnail CTR (%)', 'Views',
        'Average View Percentage (%)', 'Average View Duration', 'Watch Time (hours)',
        'Likes', 'Shares', 'New Comments', 'Like Rate (%)',
        'Subscribers', 'New Subscribers', 'Returning Viewers', 'New Viewers',
        'Estimated Revenue (USD)', 'Revenue per 1000 Views (USD)' 
    ]
    
    # Filter columns and create a copy
    df_analysis = df[columns_to_keep].copy()

    # Convert "Monday", "Tuesday" etc. to numbers 0-6
    day_mapping = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 
        'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    df_analysis['Day_Encoded'] = df_analysis['Day of Week'].map(day_mapping)

    # Define X (Inputs) and y (Target)
    # We drop 'Views' from X because that is what we are predicting!
    X = df_analysis.drop(columns=['Video Publish Time', 'Day of Week', 'Views'])
    y = df_analysis['Views']

    # Train the Random Forest Regressor
    model = RandomForestRegressor()
    model.fit(X, y)
    
    return model, X.columns

# Run the training function
model, feature_names = train_model()

# --- 2. The Website Interface ---
st.title("🔮 YouTube Views Predictor")

if model is None:
    st.error("Error: CSV file not found. Please place 'youtube_channel_real_performance_analytics.csv' in the same folder.")
else:
    st.write("Enter the video stats below to predict how many **Views** it will get.")

    # Create a list to store user inputs
    user_inputs = []
    
    # Loop through every feature the model needs
    for col in feature_names:
        # Special Dropdown for Day of Week
        if "Day_Encoded" in col:
            val = st.selectbox("Day of Week", options=[0,1,2,3,4,5,6], format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        # Number inputs for everything else
        else:
            val = st.number_input(f"Enter {col}", value=0.0)
        user_inputs.append(val)

    # --- 3. The Prediction Button ---
    if st.button("Predict Views"):
        # Convert list to numpy array
        final_input = np.array([user_inputs])
        
        # Ask model for prediction
        prediction = model.predict(final_input)
        
        # Show result
        st.success(f"📈 Predicted Views: {int(prediction[0]):,}")