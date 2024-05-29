
import pandas as pd

# Load the uploaded CSV file
file_path = '2019-2020 copy.csv'
stock_data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
stock_data.head()

def label_stock(row):
    if row['PER'] == 0 and row['PBR'] == 0 and row['ROE'] == 0 and row['DER'] == 0 and row['DPR'] == 0:
        return 'buruk'
    per_score = 'buruk' if row['PER'] > 25 else ('biasa saja' if 15 <= row['PER'] <= 25 else 'menarik')
    pbr_score = 'buruk' if row['PBR'] > 3 else ('biasa saja' if 1.5 <= row['PBR'] <= 3 else 'menarik')
    roe_score = 'buruk' if row['ROE'] < 5 else ('biasa saja' if 5 <= row['ROE'] <= 15 else 'menarik')
    der_score = 'buruk' if row['DER'] > 1 else ('biasa saja' if 0.5 <= row['DER'] <= 1 else 'menarik')
    dpr_score = 'buruk' if row['DPR'] < 25 else ('biasa saja' if 25 <= row['DPR'] <= 50 else 'menarik')

    scores = [per_score, pbr_score, roe_score, der_score, dpr_score]

    if scores.count('menarik') >= 3:
        return 'menarik'
    elif scores.count('buruk') >= 3:
        return 'buruk'
    else:
        return 'biasa saja'

def label_stock_number(row):
    if row['PER'] == 0 and row['PBR'] == 0 and row['ROE'] == 0 and row['DER'] == 0 and row['DPR'] == 0:
        return '1'
    per_score = 'buruk' if row['PER'] > 25 else ('biasa saja' if 15 <= row['PER'] <= 25 else 'menarik')
    pbr_score = 'buruk' if row['PBR'] > 3 else ('biasa saja' if 1.5 <= row['PBR'] <= 3 else 'menarik')
    roe_score = 'buruk' if row['ROE'] < 5 else ('biasa saja' if 5 <= row['ROE'] <= 15 else 'menarik')
    der_score = 'buruk' if row['DER'] > 1 else ('biasa saja' if 0.5 <= row['DER'] <= 1 else 'menarik')
    dpr_score = 'buruk' if row['DPR'] < 25 else ('biasa saja' if 25 <= row['DPR'] <= 50 else 'menarik')

    scores = [per_score, pbr_score, roe_score, der_score, dpr_score]

    if scores.count('menarik') >= 3:
        return '3'
    elif scores.count('buruk') >= 3:
        return '1'
    else:
        return '2'


# Apply the labeling function to each row
stock_data['Label'] = stock_data.apply(label_stock, axis=1)
stock_data['Label_number'] = stock_data.apply(label_stock_number, axis=1)

# Display the labeled dataframe
print(stock_data.head())

stock_data.to_csv('2019-2020-labeled.csv', index=False)