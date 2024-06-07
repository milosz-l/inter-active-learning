import numpy as np
import streamlit as st
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from src.inter_active_learning.core import experiment
from src.inter_active_learning.sampling import confidence_margin_sampling
from src.inter_active_learning.sampling import confidence_quotient_sampling
from src.inter_active_learning.sampling import entropy_sampling
from src.inter_active_learning.sampling import uncertainty_sampling

st.set_page_config(layout="wide")
RANDOM_STATE = 1234

# Title
st.title("Active Learning App")

# Sidebar for user inputs
st.sidebar.header("Experiment Configuration")

# Dataset selection
dataset = st.sidebar.selectbox("Select Dataset", ["titanic", "mnist"])
st.write(f"Chosen dataset: `{dataset}`")

# Stop criterion selection
## first: choose between: "Accuracy", "AUC"
stop_criterion_metric = st.sidebar.selectbox("Stop Criterion Metric", ["Accuracy", "AUC"])

## second: set the threshold for the stop criterion
stop_criterion_threshold = st.sidebar.number_input("Stop Criterion Threshold", min_value=0.0, max_value=1.0, value=0.9)

# Classifier selection
classifiers = {
    "KNN": KNeighborsClassifier(3),
    "Linear SVM": SVC(kernel="linear", probability=True, random_state=RANDOM_STATE),
    "RBF SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    "Gaussian Process": GaussianProcessClassifier(random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
    "Naive Bayes": GaussianNB(),
    "QDA": QuadraticDiscriminantAnalysis(),
}
selected_classifiers = st.sidebar.multiselect("Select Classifiers", list(classifiers.keys()), default=["Naive Bayes"])

# Uncertainty functions selection
uncertainty_fcs = {
    "Uncertainty": uncertainty_sampling,
    "Entropy": entropy_sampling,
    "Confidence margin": confidence_margin_sampling,
    "Confidence quotient": confidence_quotient_sampling,
}
selected_uncertainty_fcs = st.sidebar.multiselect("Select Uncertainty Functions", list(uncertainty_fcs.keys()), default=["Uncertainty"])

# Data splits using range sliders
train_active_vs_valid_test = st.sidebar.slider(label="Train+Active vs Valid+Test", min_value=0.01, max_value=0.99, value=0.8)

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
st.write(f"Using the following data splits (train, active, valid, test): `{data_splits}`")


# Number of samples per iteration
n_samples = st.sidebar.number_input("Number of Samples per Iteration", min_value=1, value=100)
st.write(f"Number of samples per iteration: `{n_samples}`")
# max_iterations = active_split * # TODO: need to fetch number of rows from specified dataset
# st.write(f"Maximum number of iterations for specified configuration: `{max_iterations}`")

show_train, show_valid, show_test = st.sidebar.columns(3)
show_train_data = show_train.checkbox("Train", value=False, help="Whether to show train data results")
show_valid_data = show_valid.checkbox("Valid", value=True, help="Whether to show valid data results")
show_test_data = show_test.checkbox("Test", value=False, help="Whether to show test data results")

# Run experiment button
if st.sidebar.button("Run Experiment"):
    # Prepare inputs for the experiment function
    selected_classifiers_dict = {key: classifiers[key] for key in selected_classifiers}
    selected_uncertainty_fcs_dict = {key: uncertainty_fcs[key] for key in selected_uncertainty_fcs}

    # Run the experiment
    results = experiment(
        data=[dataset],
        stop_criterion=lambda x: x[stop_criterion_metric] > stop_criterion_threshold,
        classifiers=selected_classifiers_dict,
        uncertainty_fcs=selected_uncertainty_fcs_dict,
        data_splits=data_splits,
        n_samples=[n_samples],
    )

    results = results.drop(columns=["data", "N sampled per iter"])

    max_highlight_columns = ["valid_Accuracy", "test_Accuracy", "valid_AUC", "test_AUC", "train_Negative Log Loss"]
    min_highlight_columns = ["train_Accuracy", "train_AUC", "valid_Negative Log Loss", "test_Negative Log Loss", "Iterations"]
    # drop columns which should not be shown
    if not show_train_data:
        cols_to_drop = ["train_Accuracy", "train_AUC", "train_Negative Log Loss"]
        results = results.drop(columns=cols_to_drop)
        max_highlight_columns = [col for col in max_highlight_columns if col not in cols_to_drop]
        min_highlight_columns = [col for col in min_highlight_columns if col not in cols_to_drop]
    if not show_valid_data:
        cols_to_drop = ["valid_Accuracy", "valid_AUC", "valid_Negative Log Loss"]
        results = results.drop(columns=cols_to_drop)
        max_highlight_columns = [col for col in max_highlight_columns if col not in cols_to_drop]
        min_highlight_columns = [col for col in min_highlight_columns if col not in cols_to_drop]
    if not show_test_data:
        cols_to_drop = ["test_Accuracy", "test_AUC", "test_Negative Log Loss"]
        results = results.drop(columns=cols_to_drop)
        max_highlight_columns = [col for col in max_highlight_columns if col not in cols_to_drop]
        min_highlight_columns = [col for col in min_highlight_columns if col not in cols_to_drop]

    # Display results
    st.write("Experiment Results")

    # Apply gradient coloring
    chosen_cmap = "YlGn"
    styled_results = results.style.background_gradient(subset=max_highlight_columns, cmap=chosen_cmap)

    for col in min_highlight_columns:
        styled_results = styled_results.background_gradient(subset=[col], cmap=chosen_cmap, gmap=-results[col])
    # styled_results = results.style.format(precision=2).bar(color="orange")
    st.dataframe(styled_results)

    st.write(
        f"""
    These values are highlighted as the `"greenest"` in the table above:
    - Maximum values for *{", ".join(max_highlight_columns)}*
    - Minimum values for *{", ".join(min_highlight_columns)}*
    """
    )
