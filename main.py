import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import norm

def preprocessing():
    data = pd.read_csv('marketing_data.csv', delimiter=';')
    X = data.drop(columns=['y', 'default'])
    y = data['y']

    # Convert target 'y' to binary numerical values
    y = y.map({'yes': 1, 'no': 0})

    # Handle class imbalance using undersampling
    rus = RandomUnderSampler(sampling_strategy=0.75, random_state=42)
    X, y = rus.fit_resample(X, y)

    # Map months to quarters and simplify job categories
    month_to_quarter = {
        'jan': 'Q1', 'feb': 'Q1', 'mar': 'Q1',
        'apr': 'Q2', 'may': 'Q2', 'jun': 'Q2',
        'jul': 'Q3', 'aug': 'Q3', 'sep': 'Q3',
        'oct': 'Q4', 'nov': 'Q4', 'dec': 'Q4'
    }
    X['quarter'] = X['month'].map(month_to_quarter)
    X = X.drop(columns=['month'])

    job_mapping = {
        'management': 'management',
        'blue-collar': 'blue-collar',
        'technician': 'technician',
        'admin.': 'admin.',
        'services': 'services',
        'retired': 'unemployed',
        'unemployed': 'unemployed',
        'self-employed': 'self-employed',
        'entrepreneur': 'self-employed',
        'housemaid': 'other',
        'student': 'other',
        'unknown': 'other'
    }
    X['job'] = X['job'].map(job_mapping)

    # One-hot encode categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    encoder = OneHotEncoder(sparse_output=False)
    categorical_encoded = encoder.fit_transform(X[categorical_cols])
    categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(categorical_cols))

    # Combine the scaled numerical and encoded categorical data
    X_final = pd.concat([pd.DataFrame(X[numerical_cols], columns=numerical_cols).reset_index(drop=True), 
                         categorical_encoded_df.reset_index(drop=True)], axis=1)
    
    # Ensure `X_final` and `y` have the same length
    assert X_final.shape[0] == y.shape[0], "Mismatch between X_final and y length"
    
    return X_final, y.reset_index(drop=True)

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

# Naive Bayes Implementation
class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.means = {}
        self.stds = {}
        self.priors = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.stds[c] = np.std(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = np.sum(np.log(norm.pdf(x, self.means[c], self.stds[c]) + 1e-9))  # Add small constant to avoid log(0)
                posterior = prior + likelihood
                posteriors.append(posterior)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)

# Logistic Regression Implementation
class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
    # Clip the values of z to prevent overflow in exp
        z = np.clip(z, -500, 500)  # Limits z to a range between -500 and 500
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return np.where(y_pred > 0.5, 1, 0)

# Multilayer Perceptron (MLP) Implementation
class MultilayerPerceptron:
    def __init__(self, hidden_size=10, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.input_weights = None
        self.hidden_weights = None

    def sigmoid(self, z):
        # Clip the values of z to prevent overflow in exp
        z = np.clip(z, -500, 500)  # Limits z to a range between -500 and 500
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.input_weights = np.random.randn(n_features, self.hidden_size)
        self.hidden_weights = np.random.randn(self.hidden_size, 1)
        for _ in range(self.epochs):
            # Forward pass
            hidden_input = np.dot(X, self.input_weights)
            hidden_output = self.sigmoid(hidden_input)
            output = self.sigmoid(np.dot(hidden_output, self.hidden_weights))
            
            # Backpropagation
            output_error = output - y.reshape(-1, 1)
            hidden_error = np.dot(output_error, self.hidden_weights.T) * hidden_output * (1 - hidden_output)
            
            self.hidden_weights -= self.lr * np.dot(hidden_output.T, output_error)
            self.input_weights -= self.lr * np.dot(X.T, hidden_error)

    def predict(self, X):
        hidden_output = self.sigmoid(np.dot(X, self.input_weights))
        output = self.sigmoid(np.dot(hidden_output, self.hidden_weights))
        return np.where(output > 0.5, 1, 0)
# Main function
def main():
    X, y = preprocessing()  # Preprocess the data and ensure alignment
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

    print(f"Shapes -> X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Naive Bayes Model
    naive_bayes = NaiveBayes()
    naive_bayes.fit(X_train, y_train)
    nb_predictions = naive_bayes.predict(X_test)
    nb_accuracy = np.mean(nb_predictions == y_test)
    print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")

    # Logistic Regression Model
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    lr_predictions = logistic_regression.predict(X_test)
    lr_accuracy = np.mean(lr_predictions == y_test)
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

    # Multilayer Perceptron Model
    mlp = MultilayerPerceptron()
    mlp.fit(X_train, y_train)
    mlp_predictions = mlp.predict(X_test)
    mlp_accuracy = np.mean(mlp_predictions == y_test)
    print(f"Multilayer Perceptron Accuracy: {mlp_accuracy:.4f}")

if __name__ == '__main__':
    main()
