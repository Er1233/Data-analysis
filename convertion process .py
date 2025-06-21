# Real Messy Dataset Example - Complete Conversion Process
# Let's work with truly messy data that looks like real-world data!

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

print("=== STEP 1: CREATE REALISTIC MESSY DATA ===")

# This is what real data often looks like - a nightmare!
messy_data = pd.DataFrame({
    'customer_id': ['CUST001', 'CUST002', 'CUST003', 'CUST004', 'CUST005', 'CUST006', 'CUST007', 'CUST008'],
    'age': [25, 45, None, '35', 52, 28, 'unknown', 41],  # Mixed types + missing
    'income': ['$45,000', '$75,000', '$55,000', None, '$95,000', '$38k', '$67,000', '$82,500'],  # Text money + missing
    'education': ['Bachelor', 'Master', 'high school', 'BACHELOR', 'PhD', np.nan, 'masters', 'High School'],  # Inconsistent case + missing
    'employment': ['Full-time', 'Part-time', 'Unemployed', 'Full-time', 'Self-employed', 'Part-time', '', 'Full-time'],  # Empty string
    'married': ['Yes', 'No', 'Y', 'N', 'YES', 'no', '1', '0'],  # Mixed Yes/No formats
    'application_date': ['2023-01-15', '2022-12-25', '01/15/2023', '2023-2-1', '15-Jan-2023', '2022/11/30', 'Jan 1, 2023', '2023-03-01'],  # Mixed date formats
    'loan_amount': ['10000', '25000', '15,000', '30k', '45000', '20K', '35,000', '12000'],  # Mixed number formats
    'credit_score': [720, 650, 580, None, 780, 620, 'N/A', 710],  # Missing + text
    'approved': [1, 0, 0, 1, 1, 0, 0, 1]  # Target variable - already clean!
})

print("Original messy data:")
print(messy_data)
print(f"\nData types:")
print(messy_data.dtypes)
print(f"\nData shape: {messy_data.shape}")

print("\n" + "="*60)
print("=== STEP 2: IDENTIFY PROBLEMS ===")

print("Problems we can see:")
print("• Age: mixed numbers/text/None")
print("• Income: dollar signs, commas, 'k' suffix")
print("• Education: inconsistent capitalization")
print("• Employment: empty strings")
print("• Married: Y/N/Yes/No/1/0 mixed")
print("• Dates: multiple formats")
print("• Loan_amount: commas, 'k' suffix")
print("• Credit_score: 'N/A' text")

# Check missing values
print(f"\nMissing values:")
print(messy_data.isnull().sum())

# Check unique values for each column
for col in messy_data.columns:
    if col != 'customer_id':
        print(f"\n{col} unique values: {messy_data[col].unique()}")

print("\n" + "="*60)
print("=== STEP 3: CLEAN DATA STEP BY STEP ===")

# Create a copy to work with
clean_data = messy_data.copy()

print("--- Cleaning Age ---")
# Convert age to numeric, handling 'unknown' and None
clean_data['age'] = clean_data['age'].replace('unknown', np.nan)
clean_data['age'] = pd.to_numeric(clean_data['age'], errors='coerce')
clean_data['age'].fillna(clean_data['age'].median(), inplace=True)
print(f"Age after cleaning: {clean_data['age'].tolist()}")

print("\n--- Cleaning Income ---")
# Remove $ signs, commas, and handle 'k' suffix
def clean_income(income_str):
    if pd.isna(income_str):
        return np.nan
    income_str = str(income_str).replace('$', '').replace(',', '')
    if income_str.endswith('k'):
        return float(income_str[:-1]) * 1000
    return float(income_str)

clean_data['income'] = clean_data['income'].apply(clean_income)
clean_data['income'].fillna(clean_data['income'].median(), inplace=True)
print(f"Income after cleaning (first 5): {clean_data['income'].head().tolist()}")

print("\n--- Cleaning Education ---")
# Standardize education levels
education_mapping = {
    'high school': 'High School',
    'BACHELOR': 'Bachelor',
    'bachelor': 'Bachelor',
    'Bachelor': 'Bachelor',
    'masters': 'Master',
    'Master': 'Master',
    'PhD': 'PhD'
}
clean_data['education'] = clean_data['education'].str.title()  # Capitalize first letters
clean_data['education'] = clean_data['education'].replace(education_mapping)
clean_data['education'].fillna('High School', inplace=True)  # Fill missing with most common
print(f"Education after cleaning: {clean_data['education'].unique()}")

print("\n--- Cleaning Employment ---")
# Handle empty strings and standardize
clean_data['employment'] = clean_data['employment'].replace('', 'Unknown')
print(f"Employment after cleaning: {clean_data['employment'].unique()}")

print("\n--- Cleaning Married ---")
# Standardize all Yes/No variations
married_mapping = {
    'Yes': 1, 'YES': 1, 'Y': 1, '1': 1,
    'No': 0, 'no': 0, 'N': 0, '0': 0
}
clean_data['married'] = clean_data['married'].map(married_mapping)
print(f"Married after cleaning: {clean_data['married'].tolist()}")

print("\n--- Cleaning Application Date ---")
# Convert all date formats to standard datetime
clean_data['application_date'] = pd.to_datetime(clean_data['application_date'], errors='coerce')
# Extract useful features from date
clean_data['application_year'] = clean_data['application_date'].dt.year
clean_data['application_month'] = clean_data['application_date'].dt.month
clean_data['days_since_application'] = (pd.Timestamp.now() - clean_data['application_date']).dt.days
print(f"Date features created: application_year, application_month, days_since_application")

print("\n--- Cleaning Loan Amount ---")
# Similar to income cleaning
def clean_loan_amount(amount_str):
    if pd.isna(amount_str):
        return np.nan
    amount_str = str(amount_str).replace(',', '')
    if amount_str.upper().endswith('K'):
        return float(amount_str[:-1]) * 1000
    return float(amount_str)

clean_data['loan_amount'] = clean_data['loan_amount'].apply(clean_loan_amount)
print(f"Loan amount after cleaning: {clean_data['loan_amount'].tolist()}")

print("\n--- Cleaning Credit Score ---")
# Handle 'N/A' and convert to numeric
clean_data['credit_score'] = clean_data['credit_score'].replace('N/A', np.nan)
clean_data['credit_score'] = pd.to_numeric(clean_data['credit_score'], errors='coerce')
clean_data['credit_score'].fillna(clean_data['credit_score'].median(), inplace=True)
print(f"Credit score after cleaning: {clean_data['credit_score'].tolist()}")

print("\n" + "="*60)
print("=== STEP 4: CONVERT TO NUMBERS FOR SKLEARN ===")

print("--- Encode Categorical Variables ---")
# Label encode education (ordered: High School < Bachelor < Master < PhD)
education_order = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
clean_data['education_encoded'] = clean_data['education'].map(education_order)

# One-hot encode employment (no natural order)
employment_dummies = pd.get_dummies(clean_data['employment'], prefix='employment')
clean_data = pd.concat([clean_data, employment_dummies], axis=1)

print("Education encoded:", clean_data['education_encoded'].tolist())
print("Employment dummies created:", employment_dummies.columns.tolist())

print("\n--- Select Final Features ---")
# Select only numeric columns for sklearn
feature_columns = [
    'age', 'income', 'education_encoded', 'married', 
    'application_year', 'application_month', 'days_since_application',
    'loan_amount', 'credit_score'
] + employment_dummies.columns.tolist()

X = clean_data[feature_columns]
y = clean_data['approved']

print(f"Final feature matrix shape: {X.shape}")
print(f"Features used: {X.columns.tolist()}")

# Verify everything is numeric
print(f"\nFinal data types:")
print(X.dtypes)

print(f"\nAny missing values left? {X.isnull().sum().sum()}")

print("\n" + "="*60)
print("=== STEP 5: USE WITH SKLEARN (SAME AS ALWAYS!) ===")

# Your exact same code from the iris example!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Predictions: {y_pred}")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n" + "="*60)
print("=== STEP 6: BEFORE vs AFTER COMPARISON ===")

print("BEFORE (messy):")
print("• Age: [25, 45, None, '35', 52, 28, 'unknown', 41]")
print("• Income: ['$45,000', '$75,000', '$55,000', None, '$95,000', '$38k', '$67,000', '$82,500']")
print("• Education: ['Bachelor', 'Master', 'high school', 'BACHELOR', 'PhD', nan, 'masters', 'High School']")

print("\nAFTER (sklearn-ready):")
print(f"• Age: {clean_data['age'].tolist()}")
print(f"• Income: {clean_data['income'].tolist()}")
print(f"• Education_encoded: {clean_data['education_encoded'].tolist()}")

print("\n" + "="*60)
print("=== KEY TAKEAWAYS ===")

print("✅ Real data is ALWAYS messy")
print("✅ Cleaning takes 80% of the time")
print("✅ Your sklearn code stays the same")
print("✅ The pattern is always: Clean → Convert → Train")

print("\nCommon cleaning steps:")
print("1. Handle missing values (fillna)")
print("2. Standardize text (str.title, replace)")
print("3. Convert money/numbers (remove $, commas)")
print("4. Map Yes/No to 1/0")
print("5. Extract features from dates")
print("6. Encode categories")
print("7. Select numeric columns")
print("8. Feed to sklearn!")