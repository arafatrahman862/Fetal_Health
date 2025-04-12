import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Must be first command
st.set_page_config(page_title="Fetal Health Classifier", layout="centered")

# Load data
df = pd.read_csv("fetal_health.csv")
X = df.drop(columns=["fetal_health"])
y = df["fetal_health"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model dictionary with preprocessing
models = {
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
    "SVM (RBF Kernel)": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
}

# Train & score
accuracies = {}
trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, preds)
    trained_models[name] = model

# Use best model
best_model_name = max(accuracies, key=accuracies.get)
best_model = trained_models[best_model_name]

# Display info
feature_info = {col: col.replace("_", " ").title() for col in X.columns}
classes = {1.0: "Normal", 2.0: "Suspect", 3.0: "Pathological"}
colors = {1.0: "#4caf50", 2.0: "#ffc107", 3.0: "#f44336"}

# Sidebar
with st.sidebar:
    st.title("🤰 Fetal Health Predictor")
    st.write("Predict fetal health condition using ML models.")
    st.markdown("**Classes**:\n- 1: Normal\n- 2: Suspect\n- 3: Pathological")
    st.markdown("---")
    st.caption("Created with ❤️ using Streamlit")

# App Header
st.markdown("<h1 style='text-align: center;'>🧬 Fetal Health Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Enter values or upload CSV for bulk predictions</p>", unsafe_allow_html=True)
st.markdown("---")

# Reset
if st.button("🔄 Reset All Fields"):
    for feature in feature_info:
        st.session_state[feature] = 0.0
    st.rerun()

# Example Input
if st.button("🎲 Try Example Input"):
    for key in feature_info:
        st.session_state[key] = round(random.uniform(float(df[key].min()), float(df[key].max())), 2)
    st.rerun()

# Inputs
st.markdown("### 🔢 Input Features")
user_inputs = {}
for feature, label in feature_info.items():
    user_inputs[feature] = st.number_input(f"{label}", step=0.1, format="%.2f", key=feature)

# Predict
if st.button("🚀 Predict"):
    input_df = pd.DataFrame([user_inputs])
    prediction = best_model.predict(input_df)[0]
    probability = best_model.predict_proba(input_df)[0]
    confidence = round(max(probability) * 100, 2)

    st.markdown(f"<h2 style='text-align: center; color: {colors[prediction]};'>Prediction: {classes[prediction]}</h2>", unsafe_allow_html=True)
    st.info(f"Confidence: {confidence}%")

    st.markdown("#### Class Probabilities")
    plt.bar([classes[1.0], classes[2.0], classes[3.0]], probability, color=list(colors.values()))
    st.pyplot(plt)

    if hasattr(best_model, "feature_importances_"):
        st.markdown("#### Feature Importances")
        importance = best_model.feature_importances_
        sorted_idx = importance.argsort()
        plt.figure()
        plt.barh([list(feature_info.values())[i] for i in sorted_idx], importance[sorted_idx], color='#4e8cff')
        st.pyplot(plt)

# CSV Upload
st.markdown("---")
st.markdown("### 📂 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    predictions = best_model.predict(batch_df)
    batch_df["Prediction"] = [classes[p] for p in predictions]
    st.dataframe(batch_df)
    st.download_button("⬇ Download Results", batch_df.to_csv(index=False), "predictions.csv")

# Model Comparison
st.markdown("---")
st.markdown("### 📊 Model Accuracy Comparison")
plt.figure(figsize=(8, 4))
plt.bar(accuracies.keys(), accuracies.values(), color=["orange", "skyblue", "green", "purple"])
plt.ylabel("Accuracy")
plt.title("Accuracy of Each Model")
plt.ylim(0.8, 1.0)
st.pyplot(plt)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: gray;'>© 2025 Fetal Health App</div>", unsafe_allow_html=True)
