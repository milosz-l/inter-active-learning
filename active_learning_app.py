import numpy as np
import streamlit as st
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from src.inter_active_learning.core import experiment
from src.inter_active_learning.sampling import confidence_margin_sampling
from src.inter_active_learning.sampling import confidence_quotient_sampling
from src.inter_active_learning.sampling import entropy_sampling
from src.inter_active_learning.sampling import uncertainty_sampling

# Title
st.title("Active Learning App")

# Sidebar for user inputs
st.sidebar.header("Experiment Configuration")

# Dataset selection
dataset = st.sidebar.selectbox("Select Dataset", ["titanic", "mnist"])

# Stop criterion
stop_criterion_accuracy = st.sidebar.slider("Stop Criterion (Accuracy)", 0.0, 1.0, 0.9)

# Classifier selection
classifiers = {"KNN": KNeighborsClassifier(3), "Naive Bayes": GaussianNB()}
selected_classifiers = st.sidebar.multiselect("Select Classifiers", list(classifiers.keys()), default=["KNN"])

# Uncertainty functions selection
uncertainty_fcs = {
    "Uncertainty": uncertainty_sampling,
    "Entropy": entropy_sampling,
    "Confidence margin": confidence_margin_sampling,
    "Confidence quotient": confidence_quotient_sampling,
}
selected_uncertainty_fcs = st.sidebar.multiselect("Select Uncertainty Functions", list(uncertainty_fcs.keys()), default=["Uncertainty"])

# Data splits using range sliders
train_active_vs_valid_test_max_value = 0.99
train_active_vs_valid_test = st.sidebar.slider(
    label="Train+Active vs Valid+Test", min_value=0.01, max_value=train_active_vs_valid_test_max_value, value=0.8
)

train_vs_active, valid_vs_test = st.sidebar.columns(2)

train_vs_active_min_value = 0.0
train_vs_active_max_value = train_active_vs_valid_test
train_vs_active_default_value = 0.125 * train_active_vs_valid_test
train_vs_active = train_vs_active.slider(
    label="Train vs Active", min_value=train_vs_active_min_value, max_value=train_vs_active_max_value, value=train_vs_active_default_value
)

valid_vs_test_min_value = train_active_vs_valid_test
valid_vs_test_max_value = 1.0
valid_vs_test_default_value = (valid_vs_test_min_value + valid_vs_test_max_value) / 2
valid_vs_test = valid_vs_test.slider(
    label="Valid vs Test", min_value=valid_vs_test_min_value, max_value=valid_vs_test_max_value, value=valid_vs_test_default_value
)

# Calculate data splits
train_split = round(train_vs_active, 2)
active_split = round(train_active_vs_valid_test - train_vs_active, 2)
valid_split = round(valid_vs_test - train_active_vs_valid_test, 2)
test_split = round(1.0 - train_split - active_split - valid_split, 2)

data_splits = np.array([train_split, active_split, valid_split, test_split])
st.write(f"Using the following data splits (train, active, valid, test): {data_splits}")

# Number of samples per iteration
n_samples = st.sidebar.number_input("Number of Samples per Iteration", min_value=1, value=100)

# Run experiment button
if st.sidebar.button("Run Experiment"):
    # Prepare inputs for the experiment function
    selected_classifiers_dict = {key: classifiers[key] for key in selected_classifiers}
    selected_uncertainty_fcs_dict = {key: uncertainty_fcs[key] for key in selected_uncertainty_fcs}

    # Run the experiment
    results = experiment(
        data=[dataset],
        stop_criterion=lambda x: x["Accuracy"] > stop_criterion_accuracy,
        classifiers=selected_classifiers_dict,
        uncertainty_fcs=selected_uncertainty_fcs_dict,
        data_splits=data_splits,
        n_samples=[n_samples],
    )

    # Display results
    st.write("Experiment Results")
    st.dataframe(results)