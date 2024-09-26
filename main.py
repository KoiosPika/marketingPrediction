import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def preprocessing():

    data = pd.read_csv('marketing_data.csv', delimiter=';')
    df = pd.DataFrame(data)

    X = data.drop(columns=['y','default'])
    y = data['y']

    rus = RandomUnderSampler(sampling_strategy=0.75, random_state=42)
    X, y = rus.fit_resample(X, y)

    df = pd.concat([X, y],axis = 1)

    month_to_quarter = {
        'jan': 'Q1', 'feb': 'Q1', 'mar': 'Q1',
        'apr': 'Q2', 'may': 'Q2', 'jun': 'Q2',
        'jul': 'Q3', 'aug': 'Q3', 'sep': 'Q3',
        'oct': 'Q4', 'nov': 'Q4', 'dec': 'Q4'
    }
    df['quarter'] = df['month'].map(month_to_quarter)

    df = df.drop(columns = ['month'])

    job_mapping = {
        'management' : 'management',
        'blue-collar' : 'blue-collar',
        'technician' : 'technician',
        'admin.' : 'admin.',
        'services' : 'services',
        'retired': 'unemployed', 
        'unemployed': 'unemployed',
        'self-employed': 'self-employed',
        'entrepreneur': 'self-employed',
        'housemaid': 'other',
        'student': 'other',
        'unknown': 'other'
    }

    df['job'] = df['job'].map(job_mapping)

    X = df.drop(columns=['y'])
    y = df['y']

    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    numerical_df = X[numerical_cols]
    categorical_df = X[categorical_cols]

    scalar = StandardScaler()
    numerical_df = scalar.fit_transform(numerical_df)
    numerical_scaled_df = pd.DataFrame(numerical_df, columns=numerical_cols)

    encoder = OneHotEncoder(sparse_output=False)
    categorical_encoded = encoder.fit_transform(categorical_df)
    categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(categorical_cols))

    categorical_scaled = scalar.fit_transform(categorical_encoded_df)
    categorical_scaled_df = pd.DataFrame(categorical_scaled, columns=encoder.get_feature_names_out(categorical_cols))

    df = pd.concat([numerical_scaled_df, categorical_scaled_df, y], axis = 1)

    X_final = df.drop(columns=['y'])
    y_final = df['y']

    return X_final, y_final

def train_test_split(X, y, test_size):
    X = np.array(X)
    y = np.array(y)
    
    np.random.seed(42)
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    split_index = int((1 - test_size) * len(indices))
    
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, y_train, X_test, y_test

class NaiveBayes:
    def __init__(self):
        # Initialize parameters like weights (theta) if needed
        self = None

    def fit(self, X, y):
        # Fit the model
        print("Training the model...")

    def predict(self, X):
        # Predict code here
        print("Predicting...")

class LogisticRegression:
    def __init__(self):
        # Initialize parameters like weights (theta) if needed
        self = None

    def fit(self, X, y):
        # Fit the model
        print("Training the model...")

    def predict(self, X):
        # Predict code here
        print("Predicting...")

class MultilayerPerceptron:
    def __init__(self):
        # Initialize parameters like weights (theta) if needed
        self = None

    def fit(self, X, y):
        # Fit the model
        print("Training the model...")

    def predict(self, X):
        # Predict code here
        print("Predicting...")   

def main():
    X, y = preprocessing()
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

    # Define Naive Bayes
    NaiveModel = NaiveBayes()
    NaiveModel.fit(X_train, y_train)
    NaivePredictions = NaiveModel.predict(X_test)

    # Measure Niave Bayes Accuracy
    correctNaive = np.sum(y_test == NaivePredictions)
    accuracyNaive = correctNaive / len(y_test)

    # Define Logistic Regression
    logisticModel = LogisticRegression()
    logisticModel.fit(X_train, y_train)
    logisticPredictions = logisticModel.predict(X_test)

    # Measure Logistic Regression Accuracy
    correctLogistic = np.sum(y_test == logisticPredictions)
    accuracyLogistic = correctLogistic / len(y_test)

    # Define Multilayer Perceptron
    MLPModel = MultilayerPerceptron()
    MLPModel.fit(X_train, y_train)
    MLPPredictions = MLPModel.predict(X_test)

    # Measure Multilayer Perceptron Accuracy
    correctMLP = np.sum(y_test == MLPPredictions)
    accuracyMLP = correctMLP / len(y_test)

if __name__ == '__main__':
    main()  