import joblib
import json
import warnings
warnings.filterwarnings("ignore")
################################################################################################
def load_modelWithScaler(model_path,scaler_path ,data ,returnName=False,dictionary = None):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    dictionary = dictionary
    data = data
    prediction = model.predict(scaler.transform([data]))
    if (returnName == True):
        for x, c in dictionary.items():
            if c == prediction:
                return x
    else:
        return prediction
################################################################################################
with open('ml_models/mapping_dict.json', 'r') as json_file: 
    mapping_dict = json.load(json_file)
################################################################################################
def ret_val_dict(col , val) : 
    return (mapping_dict[col])[val]
################################################################################################
sym_dict = mapping_dict["class"]
cat_col =  list(mapping_dict.keys())
################################################################################################
def predict_use_symptoms (sym_data):
    for i in range (1 , len(sym_data)):  
        sym_data[i] = ret_val_dict(cat_col[i-1],sym_data[i])
    pred = load_modelWithScaler(model_path="ml_models/symptoms_model.h5",scaler_path="ml_models/symptoms_scaler.h5" ,data=sym_data ,returnName=True,dictionary =sym_dict)
    return pred
################################################################################################
def cat_col_name():
    return  ["Age"]+cat_col[0:-1]
################################################################################################
