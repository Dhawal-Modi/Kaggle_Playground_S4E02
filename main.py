import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

train_data_path = "dataset/train.csv"
test_data_path = "dataset/test.csv"

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Check for missing values
missing_values = train_data.isnull().sum()

# Define categorical and numerical features
categorical_features = train_data.select_dtypes(include=['object']).drop(['NObeyesdad'], axis=1).columns.tolist()
numerical_features = train_data.select_dtypes(exclude=['object']).drop(['id'], axis=1).columns.tolist()

# Define the preprocessing for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
    ('scaler', StandardScaler())  # Standardize features by removing the mean and scaling to unit variance
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with the most frequent value
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Apply OneHotEncoder to categorical data
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model
model = RandomForestClassifier(random_state=42)

# Create the preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Split data into training and validation sets
X = train_data.drop(['id', 'NObeyesdad'], axis=1)
y = train_data['NObeyesdad']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Get score of the model
score = pipeline.score(X_val, y_val)
print(f"Model score: {score}")

# Prepare the test data (exclude the 'id' column)
X_test = test_data.drop(['id'], axis=1)

# Make predictions using the trained pipeline
predictions = pipeline.predict(X_test)

# Create a new DataFrame with the 'id' column and the predicted 'NObeyesdad' labels
submission_df = test_data[['id']].copy()  # Copy 'id' column from test data
submission_df['NObeyesdad'] = predictions  # Add the predicted labels

# Display the first few rows of the submission DataFrame
submission_df.head()

submission_df.to_csv('output/submission.csv', index=False)