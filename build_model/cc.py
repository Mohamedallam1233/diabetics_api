import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import joblib
import itertools
import json

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
def MLPredictAcc(X, y, classes , scale = False , smote = False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if (scale == True) :
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    models = {
        "XGB": XGBClassifier(),
        "KNN": KNeighborsClassifier(),
        "SVC": SVC(),
        "DT": DecisionTreeClassifier(),
        "RF": RandomForestClassifier(),
        "GaussianNB" : GaussianNB(),
        "Perceptron" : Perceptron(),
        "LinearSVC" : LinearSVC(),
        "SGDClassifier" : SGDClassifier(),
        "LogisticRegression" : LogisticRegression()
    }
    modell = []
    modell_acc = []
    model_built = {}
    for name, model in models.items():
        print(f'Training Model {name} \n--------------')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cf = confusion_matrix(y_test, y_pred)
        acc_svc = round(accuracy_score(y_test, y_pred) * 100,2)
        modell.append(name)
        modell_acc.append(acc_svc)
        model_built[name]=model
        print('-' * 30)
    models = pd.DataFrame(
        {
            'Model': modell,
            'Score': modell_acc ,

        })
    models = models.sort_values(by='Score', ascending=False)
    models['Score'] = models['Score'].apply(lambda x : str(x) + " %")
    modelss = pd.DataFrame({
        "index ": [p for p in range(1,len(modell_acc)+1)],
         "model" : models['Model'],
         'Score': models['Score'],
    })

    if (scale == True):
        return modelss, model_built , scaler
    else:
        return modelss, model_built
def check_category_classes(df):
    return df.select_dtypes(include='O').columns.to_list()
def check_non_category_classes(df):
    return df.select_dtypes(exclude='O').columns.to_list()
def define_column_type(df):
    numerical_column =check_non_category_classes(df)
    categorical_column = check_category_classes(df)
    print("numerical_column", numerical_column)
    print("categorical_column", categorical_column)
    return numerical_column , categorical_column
def make_encoding_dict(df):
    return dict(tuple(zip(df.value_counts().index.tolist(), [i for i in range (100)])))

df = pd.read_csv("diabetes_symptoms.csv")
print ("/n/n/n/n")
numerical_column , categorical_column = define_column_type(df)
df[(df["Age"]<27) | (df["Age"]>78)] = df['Age'].median()

Text_to_number = {}
for i in categorical_column : 
    Text_to_number[i] = make_encoding_dict(df[i])
    df[i]= df[i].map(make_encoding_dict(df[i]))
# Save the dictionary as a JSON file
with open('mapping_dict.json', 'w') as json_file:
    json.dump(Text_to_number, json_file)

X = df.drop('class', axis=1)
y = df['class']

models_acc, models , scaler = MLPredictAcc(X,y,['Positive' , 'Negative'],True,True)
print(models_acc)


joblib.dump(models["RF"], 'symptoms_model.h5')
joblib.dump(scaler, 'symptoms_scaler.h5')