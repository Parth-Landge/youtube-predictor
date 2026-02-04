YouTube Virality Simulator & View Predictor
A Machine Learning web application that helps content creators simulate how changes in engagement metrics (CTR, Watch Time, Likes) impact their video views.

Live Demo: [Click here to view the App(https://youtube-predictor-isfxcwkkqpl7yuk9upspnj.streamlit.app/)

Project Motive
Youtube analytics can be overwhelming. Creators often wonder: "If I had a better thumbnail, would this video have gone viral?" or "How much does watch time actually matter?"

This project isn't just a view counter; it is a "What-If" Analysis Tool. By training a Random Forest Regressor on real channel performance data, this tool allows creators to input theoretical metrics (e.g., increasing CTR by 2%) and see the predicted impact on their view count using Machine Learning.

Key Features
Virality Simulation: Predicts views based on engagement metrics like Impressions, CTR, and Retention.

Smart Preprocessing: Automatically handles categorical data (Day of Week) and cleans raw analytics data.

Interactive UI: Built with Streamlit for a user-friendly, dark-mode web interface.

Real-time Inference: Uses a pre-trained Random Forest Regressor to generate instant predictions.

Tech Stack
Language: Python

Machine Learning: Scikit-Learn (Random Forest Regressor)

Data Manipulation: Pandas, NumPy

Web Framework: Streamlit

Deployment: Streamlit Community Cloud

Limitations & Analysis
Post-Upload Metrics: The current model relies on inputs that are only available after a video is published (e.g., Likes, Watch Time, Impressions).

Implication: This tool is best used as a Scenario Simulator (analyzing past performance or theoretical targets) rather than a pre-upload fortune teller.

Data Bias: The model is trained on a specific dataset of channel performance. It may not generalize perfectly to channels in completely different niches (e.g., Gaming vs. Cooking) without re-training.

Metric Interdependence: In reality, Views and Impressions are highly correlated. The model learns this mathematical relationship, which yields high accuracy (R² Score) but reflects the platform's algorithm rather than predicting human behavior from scratch.

🔮 Future Scope
Version 2.0 (Pre-Upload Predictor): Train a separate model using only pre-upload features (Title Length, Topic, Video Duration, Day of Week) to help creators optimize content before publishing.

NLP Integration: Analyze video titles and tags to see which keywords correlate with higher views.

How to Run Locally
Clone the repository:
git clone https://github.com/Parth-Landge/youtube-predictor.git

Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run app.py

Created by Parth Landge
