import pickle

import pandas as pd
from flask import Flask, jsonify, render_template

# import json

#%% Set up environment
app = Flask(__name__)


def predict_func(id, data_pred, data, model):
    ID_toPred = data_pred[data["SK_ID_CURR"] == id]
    prediction = model.predict(ID_toPred)
    proba = model.predict_proba(ID_toPred)
    return prediction, proba[0][0]


#%% import data and model
pathToData = "pickle/DataBaseTest2.pickle"
data_pred = pd.read_pickle(pathToData)

pathToData = "pickle/testDash.pickle"
data = pd.read_pickle(pathToData).drop(
    ["Prevision", "Probas_Prevision", "Prevision_seuil_50"], axis=1
)
pathToModel = "pickle/model_Forest1.pickle"
with open(pathToModel, "rb") as model_predict:
    model = pickle.load(model_predict)

#%% create api
@app.route("/")
def home():
    return "Hello, welcome to the credit scoring api"


@app.route("/prediction/<ID>", methods=["GET"])
def predictID(ID):
    prediction, proba = predict_func(int(ID), data_pred, data, model)
    results = {"prediction": int(prediction), "proba": proba}
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)
