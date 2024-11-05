import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

st.set_page_config(
    page_title="Mushroom Classification App",
    page_icon="🍄",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .stButton>button { background-color: #b8e6d1; color: #2c3e50; border-radius: 10px; padding: 10px 24px; }
    .stButton>button:hover { background-color: #8fcbb3; transform: scale(1.05); }
    .metric-card, .sidebar .stSelectbox, .sidebar .stSlider { background-color: #ffd7e4; border-radius: 10px; padding: 10px; }
    .title-text { color: #4a4a4a; font-size: 40px; font-weight: bold; text-align: center; padding: 20px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'data', 'mushrooms.csv')
        data = pd.read_csv(file_path)
        for col in data.columns:
            data[col] = LabelEncoder().fit_transform(data[col])
        return data
    except FileNotFoundError:
        st.error("Data file not found. Please place 'mushrooms.csv' in the 'data' directory.")
        return pd.DataFrame()

@st.cache_data
def split_data(df):
    y = df['type']
    x = df.drop(columns=['type'])
    return train_test_split(x, y, test_size=0.3, random_state=0)

def plot_metrics(metrics, model, x_test, y_test):
    for metric in metrics:
        st.subheader(metric)
        fig, ax = plt.subplots()
        if metric == 'Confusion Matrix':
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, ax=ax, display_labels=['edible', 'poisonous'])
        elif metric == 'ROC Curve':
            RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
        elif metric == 'Precision-Recall Curve':
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test, ax=ax)
        st.pyplot(fig)

def main():
    st.markdown('<div class="title-text">🍄 Mushroom Classification App 🍄</div>', unsafe_allow_html=True)
    st.markdown("Is this mushroom edible or poisonous?")

    with st.spinner('🍄 Loading mushroom data...'):
        df = load_data()
    if df.empty:
        return

    st.success("Data loaded successfully! Let's start classifying mushrooms! 🎉")
    x_train, x_test, y_train, y_test = split_data(df)

    st.sidebar.header("🎮 Control Panel")
    classifier_name = st.sidebar.selectbox("🤖 Choose Your Classifier", ["Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"])

    if classifier_name == 'Support Vector Machine (SVM)':
        C = st.sidebar.number_input("C (Regularization)", 0.01, 10.0, 1.0, 0.01)
        kernel = st.sidebar.radio("Kernel", ["rbf", "linear"])
        gamma = st.sidebar.radio("Gamma", ["scale", "auto"])
        model = SVC(C=C, kernel=kernel, gamma=gamma)

    elif classifier_name == 'Logistic Regression':
        C = st.sidebar.number_input("C (Regularization)", 0.01, 10.0, 1.0, 0.01)
        max_iter = st.sidebar.slider("Max Iterations", 100, 500, 100)
        model = LogisticRegression(C=C, max_iter=max_iter)

    elif classifier_name == 'Random Forest':
        n_estimators = st.sidebar.number_input("Number of Trees", 100, 1000, 100, 10)
        max_depth = st.sidebar.number_input("Max Depth", 1, 20, 10)
        bootstrap = st.sidebar.checkbox("Bootstrap Samples", True)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)

    metrics = st.sidebar.multiselect("Select Metrics to Plot", ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"])
    if st.sidebar.button("🚀 Classify!"):
        with st.spinner("Training your model..."):
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            precision = precision_score(y_test, y_pred).round(2)
            recall = recall_score(y_test, y_pred).round(2)

            st.write(f"**Accuracy**: {accuracy:.2f}")
            st.write(f"**Precision**: {precision}")
            st.write(f"**Recall**: {recall}")
            plot_metrics(metrics, model, x_test, y_test)

    if classifier_name == 'Random Forest':
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': x_train.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        st.bar_chart(feature_importance.set_index('Feature').head(10))

    if st.sidebar.checkbox("Show Dataset"):
        st.subheader("Mushroom Dataset Explorer")
        st.write(f"Dataset Shape: {df.shape}")
        st.dataframe(df)

if __name__ == "__main__":
    main()
