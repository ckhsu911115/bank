import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle


data = pd.read_csv('train_data.csv')

# 風險等級數字化
risk_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
data['Risk_Level'] = data['Risk_Level'].map(risk_mapping)

# 就業狀況數字化
employment_mapping = {'Full-Time': 0, 'Part-Time': 1, 'Unemployed': 2}
data['Employment_Status'] = data['Employment_Status'].map(employment_mapping)

# 計算年收入與 DTI 比率
data['Gross_Annual_Income'] = data['Monthly_Salary'] * 12
data['DTI_Ratio'] = (data['Loan_Amount'] / data['Gross_Annual_Income']) * 100

data['Loan_Amount_Log'] = np.log1p(data['Loan_Amount'])

X = data[['Monthly_Salary', 'Credit_Score', 'Loan_Amount_Log', 'DTI_Ratio', 'Employment_Status', 'Assets_Value']]
y = data['Risk_Level']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
for train_index, test_index in skf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("混淆矩陣：")
    print(confusion_matrix(y_test, y_pred))
    print("分類報告：")
    print(classification_report(y_test, y_pred))

# 特徵重要性
feature_names = ['Monthly_Salary', 'Credit_Score', 'Loan_Amount_Log', 'DTI_Ratio', 'Employment_Status', 'Assets_Value']
importances = model.feature_importances_
print("特徵重要性：")
for feature, importance in zip(feature_names, importances):
    print(f"{feature}: {importance}")

with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("模型與標準化器已保存。")
