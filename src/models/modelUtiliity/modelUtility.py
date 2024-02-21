import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, classification_report, calinski_harabasz_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from capstone_group03 import clean_column_replace_nan_with_1, process_CAFV, categorize_cafv_eligibility
import numpy as np

def cleanDC(dataFrameDC):
    dataFrameDC.columns = map(str.lower, dataFrameDC.columns)
    dataFrameDC = dataFrameDC.map(lambda s: s.lower() if type(s) == str else s)
    dataFrameDC = dataFrameDC[["vin (1-10)", "county", "electric range", "electric vehicle type", "clean alternative fuel vehicle (cafv) eligibility"]]
    dataFrameDC = dataFrameDC.rename(columns={"clean alternative fuel vehicle (cafv) eligibility": "cafv eligibility"})
    dataFrameDC["cafv eligibility"] = np.where(dataFrameDC["cafv eligibility"] == "clean alternative fuel vehicle eligible", "yes", "no")
    return dataFrameDC

def process_CAFV(cafv):
  if "clean" in cafv.lower():
      return 1
  elif "eligibility unknown"  in cafv.lower():
      return 0
  return -1

def categorize_cafv_eligibility(eligibility_text):
    if "not eligible" in eligibility_text.lower():
        return 0
    else:
        return 1

def clean_column_replace_nan_with_1(dataframe, column_name):
    """
    Clean a DataFrame column by replacing NaN values with 1.

    Args:
    - dataframe: The DataFrame to be cleaned.
    - column_name: The name of the column to be cleaned.

    Returns:
    - The cleaned DataFrame.
    """
    # Use the fillna method to replace NaN values with 1 in the specified column
    dataframe[column_name] = dataframe[column_name].fillna(1)
    
    return dataframe
# Data Loading and Duplication
def load_and_duplicate_data(file_path):
    df = pd.read_csv(file_path)
    data_frames = {name: df.copy() for name in ['RC', 'DC', 'LR', 'LogReg']}
    return data_frames

# Data Cleaning
def replace_nans(data_frame, columns):
    for column in columns:
        data_frame = clean_column_replace_nan_with_1(data_frame, column)
    return data_frame

def fill_missing_values(data_frame, fill_values):
    for column, value in fill_values.items():
        data_frame[column].fillna(value, inplace=True)
    return data_frame

def drop_columns(data_frame, columns):
    data_frame.drop(columns, axis=1, inplace=True)
    return data_frame

# Exploratory Data Analysis
def plot_histogram(data_frame, column):
    plt.hist(data_frame[column])
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

def plot_bar_chart(data_frame, column, function):
    data_frame[column] = data_frame[column].apply(function)
    data_frame[column].value_counts().plot(kind="bar")
    plt.title(f"Bar Chart of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.show()

# More EDA functions can be added in a similar fashion...

# Data Preprocessing
def encode_labels(data_frame, columns):
    encoder = LabelEncoder()
    for column in columns:
        data_frame[column] = encoder.fit_transform(data_frame[column])
    return data_frame

# Model Building and Evaluation
def build_and_evaluate_decision_tree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:", report)
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

def perform_kmeans_clustering(data_frame, features):
    # Preprocessing: Label encoding and standardization
    label_encoder = LabelEncoder()
    for feature in ['Make', 'Model']:
        data_frame[feature] = label_encoder.fit_transform(data_frame[feature])

    X = data_frame[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Determine the optimal number of clusters using the Elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X_pca)
        wcss.append(kmeans.inertia_)

    # Plot the Elbow graph
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    # Perform K-Means clustering with the optimal number of clusters
    optimal_clusters = 3  # Adjust based on the Elbow graph
    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_pca)

    # Add cluster labels to the dataset and visualize
    data_frame['Cluster'] = kmeans.labels_
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.title('K-Means Clustering')
    plt.show()

def perform_logistic_regression(data_frame, features, target):
    # Preprocessing: Label encoding
    label_encoder = LabelEncoder()
    data_frame[target] = label_encoder.fit_transform(data_frame[target])

    X = data_frame[features]
    y = data_frame[target]

    # Split the dataset and fit the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log_model = LogisticRegression(solver='liblinear', random_state=42)
    log_model.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = log_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:", classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def perform_gradient_boosting(data_frame, features, target):
    # Preprocessing: Label encoding
    label_encoder = LabelEncoder()
    data_frame[target] = label_encoder.fit_transform(data_frame[target])

    X = data_frame[features]
    y = data_frame[target]

    # Split the dataset and fit the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:", classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Linear Regression
def linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

# Random Forest Classifier
def random_forest_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print_evaluation_scores(y_test, y_pred)

def print_evaluation_scores(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

# Main function to demonstrate usage
def main():
    data_frames = load_and_duplicate_data("https://data.wa.gov/api/views/f6w7-q2d2/rows.csv?accessType=DOWNLOAD")
    # Further processing and function calls...

if __name__ == "__main__":
    main()
