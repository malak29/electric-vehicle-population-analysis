from flask import Flask, request, jsonify
import pandas as pd
from utilities import (load_and_duplicate_data, replace_nans, fill_missing_values,
                       drop_columns, encode_labels, build_and_evaluate_decision_tree,
                       k_means_clustering, logistic_regression, gradient_boosting_classifier,
                       linear_regression, random_forest_classifier)

app = Flask(__name__)

@app.route('/execute_model', methods=['POST'])
def execute_model():
    data = request.json
    model_name = data['model_name']
    X = pd.DataFrame(data['X'])  # Assuming 'X' is passed as a list of dictionaries
    y = pd.Series(data['y']) if 'y' in data else None  # 'y' might not be present for some models like K-Means

    # Decision Tree
    if model_name == 'decision_tree':
        result = build_and_evaluate_decision_tree(X, y)

    # K-Means Clustering
    elif model_name == 'k_means':
        # Additional parameters for K-Means, if any, should be extracted from 'data'
        result = k_means_clustering(X)

    # Logistic Regression
    elif model_name == 'logistic_regression':
        result = logistic_regression(X, y)

    # Gradient Boosting Classifier
    elif model_name == 'gradient_boosting':
        result = gradient_boosting_classifier(X, y)

    # Linear Regression
    elif model_name == 'linear_regression':
        result = linear_regression(X, y)

    # Random Forest Classifier
    elif model_name == 'random_forest':
        result = random_forest_classifier(X, y)

    else:
        return jsonify({"error": "Model not supported"}), 400

    # Assuming 'result' is a dictionary containing the outcome of the model execution
    return jsonify({"message": "Model executed successfully", "result": result})

if __name__ == '__main__':
    app.run(debug=True)