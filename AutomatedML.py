import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Create a sample dataset for demonstration (Classification problem)
data = {
    'Feature1': np.random.rand(200),
    'Feature2': np.random.rand(200),
    'Feature3': np.random.randint(0, 100, 200),
    'Label': np.random.choice(['ClassA', 'ClassB'], size=200)
}
df = pd.DataFrame(data)

# Save dataset as CSV for demonstration purposes
df.to_csv('sample_dataset.csv', index=False)

# Load the dataset
df = pd.read_csv('sample_dataset.csv')

# Data Processing
def data_processing(df):
    print("Initial Data Info:")
    print(df.info())

    # Handling missing values
    df.fillna(method='ffill', inplace=True)

    # Encoding categorical variables
    if 'Label' in df.columns:
        label_enc = LabelEncoder()
        df['Label'] = label_enc.fit_transform(df['Label'])

    print("Processed Data Info:")
    print(df.info())
    return df

# Descriptive Analysis
def descriptive_analysis(df):
    print("\nDescriptive Statistics:")
    print(df.describe(include='all'))

    print("\nCorrelation Matrix:")
    correlation_matrix = df.corr()
    print(correlation_matrix)

    # Visualizations
    sns.pairplot(df, hue='Label')
    plt.show()

# Model Selection and Training
def model_training(df, problem_type='classification'):
    # Splitting features and labels
    X = df.drop('Label', axis=1)
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if problem_type == 'classification':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    else:
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy' if problem_type == 'classification' else 'r2')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    if problem_type == 'classification':
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    else:
        print(f"MSE: {mean_squared_error(y_test, y_pred)}")
        print(f"R2 Score: {r2_score(y_test, y_pred)}")

    print(f"Best Parameters: {grid_search.best_params_}")

    return best_model

# Main pipeline
if __name__ == "__main__":
    print(" Loading Dataset ")
    processed_df = data_processing(df)

    print(" Descriptive Analysis ")
    descriptive_analysis(processed_df)

    print(" Model Training ")
    best_model = model_training(processed_df, problem_type='classification')

    print(" Script Complete ")
