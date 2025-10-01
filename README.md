# Breast Cancer Classification with a Tuned Random Forest

This project demonstrates a complete machine learning pipeline for classifying breast cancer tumors as malignant or benign. The primary goal was to achieve the highest possible accuracy by moving from a baseline model to a highly-tuned, powerful classifier.

The final optimized model achieves an accuracy of 95.61% on the unseen test data.
Project Workflow

The project followed a systematic approach to maximize model performance:
1. Data Preparation

    Dataset: The model was trained on the Wisconsin Breast Cancer dataset, which is included in the scikit-learn library.

    Data Splitting: The data was split into an 80% training set and a 20% testing set to ensure a robust evaluation.

    Feature Scaling: StandardScaler was applied to the dataset to normalize the feature ranges. This is a crucial step that helps the model learn more effectively by preventing features with large scales from dominating the learning process.

2. Model Selection

A Random Forest Classifier was chosen as the final model. This is an ensemble model that combines multiple decision trees to produce more accurate and stable predictions than a single tree, and it is generally more powerful than simpler models like Naive Bayes.
3. Hyperparameter Tuning

To squeeze the maximum performance out of the Random Forest model, a comprehensive search for the best settings was performed using GridSearchCV. This tool automates the process of testing different model configurations.

The search grid included:

    n_estimators: [100, 200, 300]

    max_depth: [10, 20, 30, None]

    min_samples_leaf: [1, 2, 4]

    max_features: ['sqrt', 'log2']

4. Final Evaluation

The model with the best-found hyperparameters was then used to make predictions on the scaled test data, which it had never seen before.
Results

    Final Accuracy: 95.61%

    Best Hyperparameters Found:

    {
      "max_depth": 10,
      "max_features": "sqrt",
      "min_samples_leaf": 1,
      "n_estimators": 200
    }

How to Run This Project

You can replicate these results by following the steps below:

1. Clone the repository:

git clone <your-repository-url>
cd <repository-name>

2. Install the necessary libraries:

pip install scikit-learn numpy

3. Run the Jupyter Notebook:
Open the Untitled10(1).ipynb file in a Jupyter environment (like Jupyter Notebook, JupyterLab, or Google Colab) and run the cells sequentially from top to bottom.
