## MLP Benchmark: NumPy vs Keras

This project compares two Multi-Layer Perceptron (MLP) implementations—one written from scratch using NumPy, and the other using Keras—on a structured binary classification task (heart disease prediction). The project includes full preprocessing, training, evaluation, MLflow integration, and a visual dashboard.

### Features

- Custom MLP implementation using NumPy
- Keras-based MLP as a baseline
- Modular data processing pipeline with feature engineering
- MLflow tracking of metrics, parameters, and artifacts
- Interactive experiment comparison dashboard (Streamlit)
- Structured experimentation documented in Jupyter notebooks
- Built-in model caching: skips training for already existing configurations

### Experimentation

The entire experimental process is captured in the `notebooks/` directory, including:
- Baseline modeling and exploratory analysis
- Data cleaning and anomaly handling
- Feature engineering and dimensionality reduction
- Pipeline testing and model training
- Implementation comparison between NumPy and Keras

The final pipeline implementation is based directly on insights and results from these notebooks.

### Optimization

To save time and computational resources, the system checks if a model with a given configuration (backend, learning rate, epochs, etc.) has already been trained and saved. If so, it loads the existing model instead of retraining from scratch. This allows efficient iteration during experimentation.

### Project Structure

```
src/
├── models/                 # Saved model artifacts
├── models_architectures/  # MLP implementations (NumPy, Keras)
├── notebooks/             # Jupyter-based experiment tracking
├── pipelines/             # Modular pipeline strategies
├── utils/                 # Utility functions (MLflow, data, etc.)
├── mlp_manager.py         # Orchestrates full training pipeline
dashboard.py               # Streamlit-based comparison interface
Dockerfile
docker-compose.yml
```

### Running the Project

Using Docker:

```bash
docker-compose up --build
```

Once up:
- Jupyter interface is available at [localhost:8888](http://localhost:8888)
- Streamlit dashboard at [localhost:8501](http://localhost:8501)

### Running Tests

The project includes an isolated `tests` service in the Docker Compose setup. You can run the full test suite using:

```bash
docker-compose run tests
```

This will execute all tests located in the `tests/` directory using pytest.

### Dataset

The dataset is automatically downloaded from Kaggle (`fedesoriano/heart-failure-prediction`) using `kagglehub`. Preprocessing includes label encoding, frequency encoding, outlier removal, and derived feature generation.

### Model Evaluation and Tracking

All experiments are logged to MLflow, including accuracy, AUC, training duration, and hyperparameters. Models can be compared and ranked through the Streamlit dashboard, with optional cleanup of artifacts and run history.
