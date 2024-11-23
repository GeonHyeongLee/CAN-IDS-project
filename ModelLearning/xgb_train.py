from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

def hex_to_log(x):
    try:
        if isinstance(x, str):  
            return np.log1p(float(int(x, 16)))  
        else:
            return 0  
    except ValueError:
        return 0  

file_path = './datasets/1029_labeled.csv'
data = pd.read_csv(file_path)

print(data.dtypes)

data['ID'] = data['ID'].apply(hex_to_log)
data['Data'] = data['Data'].apply(hex_to_log)

features = ['Time', 'Length', 'ID', 'Data', 'Same Data', 'Entropy']  
X = data[features]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

train_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
test_dmatrix = xgb.DMatrix(data=X_test, label=y_test)

params = {
    'objective': 'multi:softmax',  
    'num_class': len(y.unique()),  
    'max_depth': 6,
    'eta': 0.1,  
    'seed': 42
}

xgb_model = xgb.train(
    params=params,
    dtrain=train_dmatrix,
    num_boost_round=100
)

y_pred = xgb_model.predict(test_dmatrix)

print(classification_report(y_test, y_pred))

model_filename = 'xgb_model.pkl'

with open(model_filename, 'wb') as file:
    pickle.dump(xgb_model, file)
print(f"Model saved to {model_filename}")

with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)
print("Model loaded successfully")

loaded_y_pred = loaded_model.predict(test_dmatrix)
print("Loaded model prediction complete")

print(classification_report(y_test, loaded_y_pred))
