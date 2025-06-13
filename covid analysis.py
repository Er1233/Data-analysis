import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def data_and_cleaning():
    df = pd.read_csv("full_data.csv")
    df = df.dropna()
    df = df[df["location"] == "Upper-middle-income countries" ]
    df = df.dropna(subset=["new_cases", "new_deaths"])

    df = df[(df["new_cases"] !=0) & (df["new_deaths"] !=0)]

    print(f"Data shape after cleaning: {df.shape}")
    return df
def create_feature(df):

    df_feature = df.copy()

    df_feature["cases_7d_avg"] = df_feature["new_cases"].rolling(7).mean()
    df_feature["deaths_7d_avg"] = df_feature["new_deaths"].rolling(7).mean()

    df_feature["target"] = (df_feature["new_cases"].shift(-1) > df_feature["new_cases"]).astype(int)

    df_feature = df_feature.dropna()
    return df_feature

def train_model(df_feature):
    feature = ["cases_7d_avg", "deaths_7d_avg"]

    x = df_feature[feature]
    y = df_feature["target"]

    x_train, x_test, y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=False)

    model = LogisticRegression(random_state=42)
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, pred)
    return model, x_test, y_test, pred, accuracy


def evaluate_and_visualize(model, X_test, y_test, y_pred, accuracy):
    """Simple evaluation and visualization"""

    print("\n" + "=" * 50)
    print("MODEL 1 RESULTS - SIMPLE BASELINE")
    print("=" * 50)
    print(f"Model Type: Logistic Regression")
    print(f"Number of Features: 2")
    print(f"Test Set Size: {len(X_test)}")
    print(f"Accuracy: {accuracy:.3f} ({accuracy * 100:.1f}%)")

    # Show model coefficients (feature importance for logistic regression)
    feature_names = X_test.columns
    coefficients = model.coef_[0]

    print(f"\nFeature Coefficients:")
    for feature, coef in zip(feature_names, coefficients):
        print(f"  {feature}: {coef:.4f}")

    # Simple visualization
    plt.figure(figsize=(12, 4))

    # Plot 1: Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.plot(y_test.values[:50], 'o-', label='Actual', alpha=0.7)
    plt.plot(y_pred[:50], 's-', label='Predicted', alpha=0.7)
    plt.title('Actual vs Predicted (First 50 test samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Direction (0=Down, 1=Up)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Feature importance
    plt.subplot(1, 2, 2)
    plt.bar(feature_names, np.abs(coefficients))
    plt.title('Feature Importance (Absolute Coefficients)')
    plt.xlabel('Features')
    plt.ylabel('Absolute Coefficient Value')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the simple baseline model"""

    print("Starting Model 1: Simple Baseline")
    print("-" * 40)

    # Step 1: Load and clean data
    df = data_and_cleaning()

    # Step 2: Create simple features
    df_features = create_feature(df)

    # Step 3: Train model
    model, X_test, y_test, y_pred, accuracy = train_model(df_features)

    # Step 4: Evaluate and visualize
    evaluate_and_visualize(model, X_test, y_test, y_pred, accuracy)

    print(f"\nKey Learning Points from Model 1:")
    print(f"1. Used only 2 simple features (rolling averages)")
    print(f"2. Logistic regression is good for binary classification")
    print(f"3. Simple train/test split for evaluation")
    print(f"4. Accuracy is a basic but important metric")
    print(f"5. Feature coefficients show feature importance")


if __name__ == "__main__":
    main()






