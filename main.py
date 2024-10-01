import pandas as pd
import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim

def preprocessing():
    data = pd.read_csv('marketing_data.csv', delimiter=';')
    X = data.drop(columns=['y', 'default'])
    y = data['y']

    y = y.map({'yes': 1, 'no': 0})

    np.random.seed(42)
    
    class_0_indices = y[y == 0].index
    class_1_indices = y[y == 1].index
    
    n_class_1 = len(class_1_indices)
    n_class_0 = int(n_class_1 / 0.83)
    
    class_0_sample_indices = np.random.choice(class_0_indices, size=n_class_0, replace=False)
    
    undersampled_indices = np.concatenate([class_0_sample_indices, class_1_indices])
    
    np.random.shuffle(undersampled_indices)
    
    X = X.loc[undersampled_indices].reset_index(drop=True)
    y = y.loc[undersampled_indices].reset_index(drop=True)

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

    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in numerical_cols:
        mean = X[col].mean()
        std = X[col].std()
        X[col] = (X[col] - mean) / std

    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

    encoded_cols = X_encoded.columns.difference(numerical_cols)
    for col in encoded_cols:
        mean = X_encoded[col].mean()
        std = X_encoded[col].std()
        X_encoded[col] = (X_encoded[col] - mean) / std

    X_final = X_encoded.reset_index(drop=True)
    y_final = y.reset_index(drop=True)

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

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
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


class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultilayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()                          
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def fit(self, X, y, epochs=1000):
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y), dtype=torch.long)

        optimizer = optim.Adam(self.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            outputs = self.forward(X_tensor)
            loss = criterion(outputs, y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, X):
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
        with torch.no_grad():
            outputs = self.forward(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.numpy()
        

def main():
    X, y = preprocessing()
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)


    naive_bayes = NaiveBayes()
    naive_bayes.fit(X_train, y_train)
    nb_predictions = naive_bayes.predict(X_test)
    nb_accuracy = np.mean(nb_predictions == y_test)
    print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")

    
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    lr_predictions = logistic_regression.predict(X_test)
    lr_accuracy = np.mean(lr_predictions == y_test)
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

    
    input_size = X_train.shape[1]
    hidden_size = 50
    output_size = len(np.unique(y_train))  
    mlp = MultilayerPerceptron(input_size, hidden_size, output_size)
    mlp.fit(X_train, y_train)
    mlp_predictions = mlp.predict(X_test)
    mlp_accuracy = np.mean(mlp_predictions == y_test)
    print(f"Multilayer Perceptron Accuracy: {mlp_accuracy:.4f}")

if __name__ == '__main__':
    main()