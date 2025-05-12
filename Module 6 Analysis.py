# Module 6 Analysis

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("student_data.csv")

# Create binary target variable for high weekend alcohol use
df['high_alc_use'] = (df['Walc'] >= 4).astype(int)

# Select academic and familial features
features = ['G1', 'G2', 'studytime', 'failures', 'absences',
            'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob',
            'guardian', 'famsup', 'schoolsup']

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df[features + ['high_alc_use']], drop_first=True)

# Define features (X) and label (y)
X = df_encoded.drop(columns='high_alc_use')
y = df_encoded['high_alc_use']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Identify misclassified samples
misclassified_indices = np.where(y_pred != y_test)[0]
misclassified_samples = X_test.iloc[misclassified_indices].copy()
misclassified_samples['Actual'] = y_test.iloc[misclassified_indices].values
misclassified_samples['Predicted'] = y_pred[misclassified_indices]

# Show 5 misclassified samples
print("5 Misclassified Samples:")
print(misclassified_samples.head(5))

