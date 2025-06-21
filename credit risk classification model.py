import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("UCI_Credit_Card.csv")

print("STEP 1: UNDERSTAND YOUR DATA")
print("="*40)
print(f"Data shape: {df.shape}")
print(f"\nTarget variable (what we're predicting):")
print(df["default.payment.next.month"].value_counts())
print(f"Default rate: {df['default.payment.next.month'].mean():.2%}")

# STEP 2: BASIC DATA CLEANING (Simple rules to follow)
print("\nSTEP 2: BASIC DATA CLEANING")
print("="*40)

# Rule 1: Remove ID columns (they don't predict anything)
columns_to_remove = ['ID']  # Add any ID-like columns here
df_clean = df.drop(columns_to_remove, axis=1)
print(f"Removed columns: {columns_to_remove}")

# Rule 2: Check for missing values
print(f"Missing values: {df_clean.isnull().sum().sum()}")

# Rule 3: Look at your categorical variables (integers that represent categories)
categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']  # These are categories, not numbers
print(f"\nCategorical variables:")
for col in categorical_cols:
    print(f"{col}: {sorted(df_clean[col].unique())}")

# STEP 3: SIMPLE PREPROCESSING (Just the essentials)
print("\nSTEP 3: SIMPLE PREPROCESSING")
print("="*40)

# For this dataset, we'll keep it simple - no complex feature engineering
# Just basic cleaning of obvious data issues

# Clean EDUCATION: group unusual values (0,5,6) with 'others' (4)
df_clean['EDUCATION'] = df_clean['EDUCATION'].replace({0: 4, 5: 4, 6: 4})

# Clean MARRIAGE: group 0 with 'others' (3)
df_clean['MARRIAGE'] = df_clean['MARRIAGE'].replace({0: 3})

print("✓ Cleaned categorical variables")

# STEP 4: SEPARATE FEATURES AND TARGET
print("\nSTEP 4: PREPARE FOR MODELING")
print("="*40)

# Separate X (features) and y (target)
X = df_clean.drop("default.payment.next.month", axis=1)
y = df_clean["default.payment.next.month"]

print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")

# STEP 5: SPLIT DATA
print("\nSTEP 5: SPLIT DATA")
print("="*40)

# Use stratify to keep same proportion of defaults in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Train default rate: {y_train.mean():.2%}")
print(f"Test default rate: {y_test.mean():.2%}")

# STEP 6: SCALE NUMERICAL FEATURES (Important for many algorithms)
print("\nSTEP 6: SCALE NUMERICAL FEATURES")
print("="*40)

# Identify numerical columns (amounts, age, limits)
numerical_cols = ['LIMIT_BAL', 'AGE'] + [f'BILL_AMT{i}' for i in range(1, 7)] + [f'PAY_AMT{i}' for i in range(1, 7)]

# Scale only numerical features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

print(f"✓ Scaled {len(numerical_cols)} numerical columns")

# STEP 7: TRAIN MODEL WITH CLASS IMBALANCE HANDLING
print("\nSTEP 7: TRAIN MODEL")
print("="*40)

# RandomForest with class_weight='balanced' handles imbalance automatically
model = RandomForestClassifier(
    n_estimators=100,           # Number of trees
    random_state=42,           # For reproducible results
    class_weight='balanced'    # Automatically handles class imbalance
)

# Fit the model
model.fit(X_train_scaled, y_train)
print("✓ Model trained successfully")

# STEP 8: MAKE PREDICTIONS
print("\nSTEP 8: MAKE PREDICTIONS")
print("="*40)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability of default

print("✓ Predictions made")

# STEP 9: EVALUATE MODEL (The metrics that matter for credit risk)
print("\nSTEP 9: EVALUATE MODEL")
print("="*40)

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.3f}")
print(f"ROC-AUC: {roc_auc:.3f}")  # This is more important than accuracy for imbalanced data

# Confusion matrix (shows what the model got right/wrong)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix:")
print(f"                 Predicted")
print(f"Actual    No Default  Default")
print(f"No Default    {tn:4d}     {fp:4d}")
print(f"Default       {fn:4d}     {tp:4d}")

# Key business metrics
print(f"\nBusiness Metrics:")
print(f"• Defaults correctly identified: {tp}/{tp+fn} ({tp/(tp+fn):.1%})")
print(f"• False alarms: {fp}/{tn+fp} ({fp/(tn+fp):.1%})")

# Classification report
print(f"\nDetailed Report:")
print(classification_report(y_test, y_pred))

# STEP 10: FEATURE IMPORTANCE (What matters most?)
print("\nSTEP 10: WHAT FEATURES MATTER MOST?")
print("="*40)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Top 10 most important features:")
for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
    print(f"{i:2d}. {row['Feature']:20s} {row['Importance']:.3f}")

print("\n" + "="*50)
print("SIMPLE RULES FOR PREPROCESSING:")
print("="*50)
print("1. Remove ID columns")
print("2. Check for missing values")
print("3. Clean categorical variables (group unusual values)")
print("4. Scale numerical features")
print("5. Use stratified split")
print("6. Handle class imbalance with class_weight='balanced'")
print("7. Use ROC-AUC, not just accuracy")
print("8. Look at confusion matrix")
print("9. Check feature importance")