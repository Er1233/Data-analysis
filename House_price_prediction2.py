from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("housing.csv")
print(df.shape)
print("Datatype", df.dtypes)
print("\n Column: ")
print(df.columns)
print(df.isnull().sum())

#future engineering

df["rooms_per_households"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["bedrooms_per_households"] = df["total_bedrooms"] / df["households"]
df["population_per_households"] = df["population"] / df["households"]

#handling missing values

imputer = SimpleImputer(strategy="median")
numerical_col = df.select_dtypes(include=[np.number]).columns
df[numerical_col] = imputer.fit_transform(df[numerical_col])

#encode categorical variables
df_encoded = pd.get_dummies(df, columns=["ocean_proximity"], prefix="ocean", drop_first=True)

#prepare x and y
x = df_encoded.drop("median_income", axis=1)

y = df_encoded["median_income"]

#log transform target
print(f"Log transform target: {y.skew():.2f}")
if y.skew() > 1:
    y = np.log(y)
    print("Applied log transformation to target")

#train test split
x_train, x_test,y_train, y_test = train_test_split(x,y, random_state=42, test_size=0.2, shuffle=True)

#feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit(x_test)

#convert back to dataframe for easier handling
x_train_scaled = pd.DataFrame(x_train_scaled, columns=x.columns, index=x_train.index)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=x.columns, index=x_test.index)

#model training with better hyperparameters

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=1
)
#model fit
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)

#calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean squared error: {mse}")
print(f"r2 score: {r2}")
print(f"mean absolute error: {mae}")

print(f"\n=== SAMPLE PREDICTIONS ===")
for i in range(3):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    error = abs(actual - predicted)
    error_pct = (error / actual) * 100

    print(f"Sample {i + 1}:")
    print(f"  Actual: ${actual:,.2f}")
    print(f"  Predicted: ${predicted:,.2f}")
    print(f"  Error: ${error:,.2f} ({error_pct:.1f}%)")
    print()



