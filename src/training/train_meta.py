import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def main():

    CSV_PATH = "data/processed/oof_meta_dataset.csv"
    df = pd.read_csv(CSV_PATH)

    y = df['targets']
    X = df.drop(columns= ['targets','image','lesion_id'])

    X = pd.get_dummies(X,columns=['sex' , 'anatom_site_general'])

    X_train , X_test , y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y) 


    model = xgb.XGBClassifier(
        use_label_encoder = False,
        eval_metric = 'mlogloss',
        random_state = 42,
        learning_rate = 0.1,
        max_depth = 4
    )
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['0_mel', '1_nv', '2_bcc', '3_ak', '4_bkl', '5_df', '6_vasc', '7_scc']))
    os.makedirs("models", exist_ok=True)
    model.save_model("models/xgb_meta_learner.json")
    column = list(X.columns)
    joblib.dump(column, "models/xgb_features.pkl")

if __name__ == "__main__":
    main()