# Import necessary libraries
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

# We will load and split the dataset
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=42)

# print(cancer)

# We will define base models
base_models = [
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
]

# We will define the meta-model
meta_model = LogisticRegression()

# We will implement Stacking
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)
stacking_model.fit(X_train, y_train)

# We will predict using the stacking model
y_pred = stacking_model.predict(X_test)

# print(y_pred[0:5])

# We will evaluate the model
accuracy = accuracy_score(y_test, y_pred)   
# print("Accuracy: {:.2f}%".format(accuracy * 100))

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
print(saw_df[['kode', 'SAW Score']])