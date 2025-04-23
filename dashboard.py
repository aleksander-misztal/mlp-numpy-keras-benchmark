import streamlit as st
import time
import shutil
import os
from src.mlp_manager import MLPManager
from src.utils.mlflow_utils import MLflowUtils

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="MLP Comparison", layout="centered")

# --- Hide default Streamlit UI elements for cleaner appearance ---
hide_streamlit_style = """
    <style>
    [data-testid="stDecoration"] {display: none;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- App Header ---
st.title("Aleksander Misztal")
st.subheader("MLP Comparison: Numpy vs Keras")
st.markdown("""
This app compares the performance of Multi-Layer Perceptrons (MLPs) implemented in **NumPy** and **Keras** on the heart disease dataset.
""")

def display_model_ranking():
    """
    Display top-performing models from MLflow, ranked by accuracy.
    """
    st.subheader("🏆 Model Ranking (MLflow)")
    ranking_df = MLflowUtils.get_model_ranking(metric="accuracy", top_n=10)

    if ranking_df.empty:
        st.warning("No models found in MLflow yet.")
    else:
        st.dataframe(ranking_df.style.format({
            "Accuracy": "{:.4f}",
            "ROC AUC": "{:.4f}",
            "Training Time (s)": "{:.2f}"
        }))

# --- Sidebar: Experiment Configuration ---
st.sidebar.header("Experiment Settings")

model_name = st.sidebar.selectbox("Select model", ["numpy", "keras"])
lr = st.sidebar.selectbox("Learning rate", [0.0001, 0.001, 0.005])
batch = st.sidebar.selectbox("Batch size", [8, 16, 32, 64])
epochs = st.sidebar.selectbox("Epochs", [20, 50, 100, 200, 300])
dropout = st.sidebar.selectbox("Dropout", [0.0, 0.1, 0.2, 0.3])

# --- Model Training Trigger ---
if st.sidebar.button("Train"):
    with st.spinner("Training in progress... Please wait ⏳"):
        st.info("Initializing training...")

        try:
            start_time = time.time()
            manager = MLPManager(model=model_name, lr=lr, batch=batch, epochs=epochs, dropout=dropout)
            manager.run()
            duration = time.time() - start_time
            results = manager.get_results()
        except Exception as e:
            st.error(f"An error occurred: {e}")
        else:
            st.success("Training finished ✅")

            # --- Results Display ---
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Confusion Matrix")
                st.pyplot(results["confusion_matrix_plot"])

                st.subheader("Classification Report")
                st.dataframe(results["classification_report"])

            with col2:
                st.subheader("ROC Curve")
                st.pyplot(results["roc_curve_plot"])

                st.metric("Accuracy", f"{results['accuracy']:.4f}")
                st.metric("ROC AUC", f"{results['roc_auc']:.4f}")
                st.metric("Training Time (s)", f"{duration:.2f}")

# --- Dangerous Operation: MLflow Cleanup ---
st.markdown("---")
st.subheader("🧹 MLflow Cleanup")

with st.expander("Danger zone: delete all MLflow runs and saved models"):
    confirm = st.checkbox("Yes, I want to delete all MLflow runs and models", key="delete_confirm")
    delete_clicked = st.button("Delete mlruns content", key="delete_button")

    if delete_clicked and confirm:
        try:
            mlruns_path = "mlruns"
            models_path = "src/models"

            # --- Delete MLflow runs ---
            if os.path.exists(mlruns_path):
                contents = [f for f in os.listdir(mlruns_path) if not f.startswith(".")]

                if not contents:
                    st.info("mlruns folder is already empty (or only contains system files).")
                else:
                    for item in contents:
                        item_path = os.path.join(mlruns_path, item)
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                    st.success("All MLflow run content deleted (excluding system files).")
            else:
                st.info("mlruns folder does not exist.")

            # --- Delete saved model files ---
            if os.path.exists(models_path):
                for file in os.listdir(models_path):
                    file_path = os.path.join(models_path, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                st.success("All saved models in src/models were deleted.")
            else:
                st.warning("src/models folder not found. No model files removed.")

        except Exception as e:
            st.error(f"Error while deleting MLflow or model files: {e}")
    elif delete_clicked:
        st.warning("Please confirm the deletion using the checkbox.")

# --- Manual Refresh for Leaderboard ---
st.markdown("---")
if st.button("🔄 Refresh ranking", key="refresh_ranking"):
    display_model_ranking()
