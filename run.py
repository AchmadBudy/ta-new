# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import ahpy

comparisons = {
    ('PBR', 'PBR'): 1,
    ('PBR', 'PER'): 2,
    ('PBR', 'DER'): 3,
    ('PBR', 'ROE'): 4,
    ('PBR', 'DPR'): 5,
    ('PER', 'PBR'): 1/2,
    ('PER', 'PER'): 1,
    ('PER', 'DER'): 2,
    ('PER', 'ROE'): 3,
    ('PER', 'DPR'): 4,
    ('DER', 'PBR'): 1/3,
    ('DER', 'PER'): 1/2,
    ('DER', 'DER'): 1,
    ('DER', 'ROE'): 2,
    ('DER', 'DPR'): 3,
    ('ROE', 'PBR'): 1/4,
    ('ROE', 'PER'): 1/3,
    ('ROE', 'DER'): 1/2,
    ('ROE', 'ROE'): 1,
    ('ROE', 'DPR'): 2,
    ('DPR', 'PBR'): 1/5,
    ('DPR', 'PER'): 1/4,
    ('DPR', 'DER'): 1/3,
    ('DPR', 'ROE'): 1/2,
    ('DPR', 'DPR'): 1
}


drinks = ahpy.Compare(name='Drinks', comparisons=comparisons, precision=3, random_index='saaty')
weights = drinks.target_weights


# PBR (Price to Book Value): Benefit
# PER (Price to Earnings Ratio): Benefit
# DER (Debt to Equity Ratio): Cost
# ROE (Return on Equity): Benefit
# DPR (Dividend Payout Ratio): Benefit

# listBenefit = ['PBR', 'PER', 'ROE', 'DPR']

# get data from csv
import pandas as pd
import numpy as np


data = pd.read_csv('2020.csv')
newData = data.copy()[["kode","PBR","PER","ROE","DER","DPR"]]

newData['PBR'] = newData['PBR'].astype(float)
newData['PER'] = newData['PER'].astype(float)   
newData['ROE'] = newData['ROE'].astype(float)
newData['DER'] = newData['DER'].astype(float)
newData['DPR'] = newData['DPR'].astype(float)

# Normalisasi
def normalize(df):
    norm_df = df.copy()
    for column in df.columns[1:]:
        column_max = norm_df[column].max()
        if column_max > 0:
            norm_df[column] = norm_df[column] / column_max
    return norm_df

# Data yang sudah dinormalisasi
normalized_df = normalize(newData.copy())

# Perhitungan SAW
def calculate_saw(df, weights):
    saw_scores = np.zeros(len(df))
    for column in df.columns[1:]:
        saw_scores += df[column] * weights[column]
    df['Target'] = saw_scores
    return df

# Data dengan nilai SAW
saw_df = calculate_saw(normalized_df.copy(), weights)

# Mengurutkan berdasarkan nilai SAW tertinggi
saw_df = saw_df.sort_values(by='Target', ascending=False)

# Menampilkan hasil akhir
# print(saw_df[['kode', 'Target']])

data = saw_df.drop(columns=['kode','Target']).values
target = saw_df['Target'].values
# print(data)
# print(target)
# X_train, X_test, y_train, y_test = train_test_split(saw_df.drop(columns=['kode','Target']), saw_df['Target'], test_size=0.2, random_state=42)
# cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Import libraries
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Definisikan base models
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=100, random_state=42)),
    ('svr', SVR())
]

# Definisikan meta-model
meta_model = LinearRegression()

# Implementasi Stacking Regressor
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Fit model pada data training
stacking_model.fit(X_train, y_train)

y_pred = stacking_model.predict(X_test)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Evaluasi kinerja model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Cetak hasil evaluasi
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# add prediction to dataframe
saw_df['Prediction'] = stacking_model.predict(data)
print(saw_df[['kode', 'Prediction','Target']])
saw_df = saw_df.sort_values(by='Prediction', ascending=False)
print(saw_df[['kode', 'Prediction','Target']])


import streamlit as st

st.title('Stock Classification System')

# show raw data

option = st.selectbox(
    "Pilih Tahun DataSet?",
    ("2019", "2020", "2021"))

st.write("You selected:", option)
st.write('Raw Data')

selectedData = pd.read_csv(f'{option}.csv')[['kode','tahun', 'PBR', 'PER', 'ROE', 'DER', 'DPR']]
st.dataframe(selectedData, use_container_width=True)

# show data table
st.write('Data set 2019 with Target Value')
rank = saw_df[['kode', 'Target']].sort_values(by='Target', ascending=False)
# fix index
rank.index = range(1,len(rank)+1)
st.dataframe(rank, use_container_width=True)

# show prediction table
st.write('classification based on 2020 data')
rankpred = saw_df[['kode', 'Prediction']].sort_values(by='Prediction', ascending=False)
# fix index
rankpred.index = range(1,len(rankpred)+1)
st.dataframe(rankpred, use_container_width=True)