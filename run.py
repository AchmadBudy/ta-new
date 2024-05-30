# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# input data training
data = pd.read_csv('2019-2020-labeled.csv')[["kode","PBR","PER","ROE","DER","DPR","Label_number"]]
print(data.head())
target = data['Label_number'].values
data = data[["PBR","PER","ROE","DER","DPR"]].values
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Import libraries
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from xgboost import XGBClassifier

# Definisikan base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, random_state=42)),
    ('svr', SVC())
]

# Definisikan meta-model
meta_model = RidgeClassifier()

# Implementasi Stacking Regressor
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

# Fit model pada data training
stacking_model.fit(X_train, y_train)

y_pred = stacking_model.predict(X_test)



data2021 = pd.read_csv('2021.csv')[["kode","PBR","PER","ROE","DER","DPR"]]
data2021WithoutKode = data2021[["PBR","PER","ROE","DER","DPR"]].values

# add prediction to dataframe
data2021['Prediction'] = stacking_model.predict(data2021WithoutKode)


import streamlit as st

st.title('Stock Classification System')

# show raw data

option = st.selectbox(
    "Pilih Untuk Melihat Data",
    ("2019", "2020", "2021"))

st.write("You selected:", option)
st.write('Raw Data')

selectedData = pd.read_csv(f'{option}.csv')[['kode','tahun', 'PBR', 'PER', 'ROE', 'DER', 'DPR']]
st.dataframe(selectedData, use_container_width=True)

# show data table
st.write('Data set 2019-2020 with Target Value')
st.dataframe(pd.read_csv('2019-2020-labeled.csv')[["kode","PBR","PER","ROE","DER","DPR","Label","Label_number"]], use_container_width=True)

# show prediction table
st.write('classification based on 2021 data')
rankpred = data2021[['kode', 'Prediction']].sort_values(by='Prediction', ascending=False)
# fix index
rankpred.index = range(1,len(rankpred)+1)
st.dataframe(rankpred, use_container_width=True)

from sklearn.metrics import classification_report
# uji akurasi
# st.write('Accuracy Ensemble Model')
print('Accuracy Ensemble Model')
print(classification_report(y_test, y_pred))

# print(y_train)
# test random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print('Accuracy Random Forest')
print(classification_report(y_test, y_pred_rf))

# test xgboost
xgb = XGBClassifier(n_estimators=100, random_state=42)
y_train_adjusted = y_train - 1
y_test_adjusted = y_test - 1
xgb.fit(X_train, y_train_adjusted)
y_pred_xgb = xgb.predict(X_test)
print('Accuracy XGBoost')
print(classification_report(y_test_adjusted, y_pred_xgb))

# test svc
svc = SVC()
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
print('Accuracy SVC')
print(classification_report(y_test, y_pred_svc))

print(y_test_adjusted)
