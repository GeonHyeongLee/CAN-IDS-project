from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
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

cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass',
    random_seed=42,
    verbose=200
)

xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',  
    num_class=len(y.unique()),
    max_depth=6,
    learning_rate=0.1,
    n_estimators=500,
    random_state=42
)

cat_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

ensemble_model = VotingClassifier(
    estimators=[('cat', cat_model), ('xgb', xgb_model)],
    voting='soft'  
)

ensemble_model.fit(X_train, y_train)

y_pred = ensemble_model.predict(X_test)
print("Ensemble Model Evaluation:")
print(classification_report(y_test, y_pred))

model_filename = 'ensem_cb_xgb_model.pkl'

with open(model_filename, 'wb') as file:
    pickle.dump(ensemble_model, file)
print(f"Ensemble model saved to {model_filename}")

with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)
print("Ensemble model loaded successfully")

loaded_y_pred = loaded_model.predict(X_test)
print("Loaded model prediction complete")

print("Classification report for loaded model:")
print(classification_report(y_test, loaded_y_pred))
