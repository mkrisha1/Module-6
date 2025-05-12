import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("student_data.csv")

# Create binary target for high weekend alcohol use
df['high_alc_use'] = (df['Walc'] >= 4).astype(int)

# Select specified features
selected_features = [
    'sex', 'age', 'famsize', 'Pstatus', 'Fedu', 'Medu',
    'Mjob', 'Fjob', 'studytime', 'failures', 'goout',
    'absences', 'Dalc', 'Walc'  # Walc is still used here to engineer the label, but should be dropped from X
]

# One-hot encode categorical features
df_encoded = pd.get_dummies(df[selected_features + ['high_alc_use']], drop_first=True)

# Drop 'Walc' from features to prevent leakage
X = df_encoded.drop(columns=['high_alc_use', 'Walc'])  
y = df_encoded['high_alc_use']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model performance
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
