from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import pickle

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

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

model_filename = 'rf_model.pkl'

with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f"Model saved to {model_filename}")

with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)
print("Model loaded successfully")

loaded_y_pred = loaded_model.predict(X_test)
print("Loaded model prediction complete")

print(classification_report(y_test, loaded_y_pred))
