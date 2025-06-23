import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("UCI_Credit_Card.csv")
print(f"Columns: {df.columns}")
print(f"\nData types: {df.dtypes}")
print(f"Missing values: {df.isnull().sum().sum()}")

#define features explicitly

categorical_features = ["SEX",'EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5',"PAY_6"]
numerical_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                     'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

#clean data
df_clean = df.drop("ID", axis=1)

#fix weird categorical values
df_clean["EDUCATION"] = df_clean["EDUCATION"].replace([0,5,6],4)
df_clean["MARRIAGE"] = df_clean["MARRIAGE"].replace(0,3)

#prepare data
x = df_clean.drop("default.payment.next.month", axis=1)
y = df_clean["default.payment.next.month"]

#split data
x_train,x_test,y_train,y_test = train_test_split(x,y, random_state=42,test_size=0.2,stratify=y)

# scale only numerical columns
scaler = StandardScaler()
x_train_scaler = x_train.copy()
x_test_scaler = x_test.copy()

x_train_scaler[numerical_features] = scaler.fit_transform(x_train[numerical_features])
x_test_scaler[numerical_features] = scaler.transform(x_test[numerical_features])

#train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
model.fit(x_train_scaler,y_train)

#evaluate
y_pred = model.predict(x_test_scaler)
y_pred_proba = model.predict_proba(x_test_scaler)[:,1]

accuracy = accuracy_score(y_test,y_pred)
roc_auc = roc_auc_score(y_test,y_pred_proba)

print(f"Accuracy: {accuracy:.3f}")
print(f"Roc-Auc: {roc_auc:.3f}")

