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

# print(drinks.target_weights)
weights = drinks.target_weights

# # normalisasi bobot
# total = sum(weights.values())
# for key in weights:
#     weights[key] = weights[key] / total

# print(weights)

# print(drinks.consistency_ratio)


# PBR (Price to Book Value): Benefit
# PER (Price to Earnings Ratio): Benefit
# DER (Debt to Equity Ratio): Cost
# ROE (Return on Equity): Benefit
# DPR (Dividend Payout Ratio): Benefit

listBenefit = ['PBR', 'PER', 'ROE', 'DPR']

def SAW(data, weights):
    # Normalisasi matriks keputusan
    normalized_data = data.copy()
    for column in data.columns[1:]:  # Kolom pertama adalah kode alternatif
        max_value = data[column].max()
        normalized_data[column] = data[column] / max_value

    # Hitung nilai total SAW untuk setiap alternatif
    saw_values = {}
    for index, row in normalized_data.iterrows():
        alternatif = row['kode']
        saw_values[alternatif] = sum(row[kriteria] * weight for kriteria, weight in zip(data.columns[1:], weights))

    return saw_values

# get data from csv
import pandas as pd
import numpy as np


data = pd.read_csv('datanya.csv')
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
    df['SAW Score'] = saw_scores
    return df

# Data dengan nilai SAW
saw_df = calculate_saw(normalized_df.copy(), weights)

# Mengurutkan berdasarkan nilai SAW tertinggi
saw_df = saw_df.sort_values(by='SAW Score', ascending=False)

# Menampilkan hasil akhir
# print(saw_df[['kode', 'SAW Score']])

data = saw_df.drop(columns=['kode','SAW Score']).values
target = saw_df['SAW Score'].values
# print(data)
# print(target)
# X_train, X_test, y_train, y_test = train_test_split(saw_df.drop(columns=['kode','SAW Score']), saw_df['SAW Score'], test_size=0.2, random_state=42)
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
print(saw_df[['kode', 'Prediction','SAW Score']])
saw_df = saw_df.sort_values(by='Prediction', ascending=False)
print(saw_df[['kode', 'Prediction','SAW Score']])


import streamlit as st

st.title('Stock Recommendation System')

st.write('This is a simple stock recommendation system using AHP and SAW method, combined with Ensemble Learning')

# show raw data
st.write('Raw Data')
st.dataframe(newData, use_container_width=True)

# show data table
st.write('Data 2019 Ranking Stock')
rank = saw_df[['kode', 'SAW Score']].sort_values(by='SAW Score', ascending=False)
# fix index
rank.index = range(1,len(rank)+1)
st.dataframe(rank, use_container_width=True)

# show prediction table
st.write('Prediction 2019 Ranking Stock')
rankpred = saw_df[['kode', 'Prediction']].sort_values(by='Prediction', ascending=False)
# fix index
rankpred.index = range(1,len(rankpred)+1)
st.dataframe(rankpred, use_container_width=True)