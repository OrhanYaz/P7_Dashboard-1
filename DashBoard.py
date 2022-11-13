import json  # ajouter au requirement
import pickle
from urllib.request import urlopen  # ajouter au requirement

import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

# Importation de la base Test
fic_sav_testDash = "pickle/testDash.pickle"
# Chargement de test_set2
with open(fic_sav_testDash, "rb") as testDash:
    TestDash = pickle.load(testDash)
TestDash.head()

# Importation de la base Test
fic_sav_trainDash = "pickle/trainDash.pickle"
# Chargement de test_set2
with open(fic_sav_trainDash, "rb") as trainDash:
    TrainDash = pickle.load(trainDash)
TrainDash.head()


# st.subheader('Platform wise sales')
# drop down for unique value from a column
platform_name = st.sidebar.selectbox("Select ID client", options=TestDash.SK_ID_CURR.unique())
API_url = "http://127.0.0.1:5000/prediction/" + str(platform_name)
json_url = urlopen(API_url)
API_data = json.loads(json_url.read())

TestDash["Probas_Prevision"] = API_data["proba"]
B = TestDash["Probas_Prevision"][TestDash.SK_ID_CURR == platform_name].values[0]


st.markdown(
    "<h1 style='text-align: center; color: red; font-size: 100px;'>DEMANDE DE PRÊT BANCAIRE</h1>",
    unsafe_allow_html=True,
)

# st.write('''BANCAIRE''')

# st.title () : écriture des titres
st.markdown(
    "<h1 style='text-align: center; color: black; font-size: 40px;'>**♟**Crédit Score du Client **♟**</h1>",
    unsafe_allow_html=True,
)

# st.title("**♟**Crédit Score du Client **♟**")


st.title("" + str(round(B, 2)) + "")

import pandas as pd
import plotly.express as px

# st.markdown("<h1 style='text-align: center; color: red;'>Some title</h1>", unsafe_allow_html=True)


if B <= 0.50:
    decision = "<font color='green'>**Acceptation de la demande de prêt pour le client**</font>"
else:
    decision = "<font color='red'>**Refus de la demande de prêt pour le client**</font>"

st.write("**Decision** *(Seuil de 50%)* **: **", decision, platform_name, unsafe_allow_html=True)

# st.write(" Acceptation de la demande de prêt pour le client", platform_name)

st.markdown(
    "<h1 style='text-align: center; color: Black; font-size: 50px;'>CARACTERISQUES DU CLIENT</h1>",
    unsafe_allow_html=True,
)
# st.header('Caracterisques du client')
C = TestDash[TestDash.SK_ID_CURR == platform_name]
C


# st.markdown('Before we get started with the analysis, lets have a quick look at the raw data :sunglasses:')


# analysis = st.sidebar.selectbox ('Select ID client', ['Image Classification', 'Data Analysis & Visualization'])

st.markdown(
    "<h1 style='text-align: center; color: Black; font-size: 50px;'>INDICATEURS INFLUENCANTS LE TAUX DE RISQUE</h1>",
    unsafe_allow_html=True,
)


import dill

# Importation de la base Test
fic_sav_Explainer = "pickle/Explainer.dat"
with open(fic_sav_Explainer, "rb") as f1:
    Explainer = dill.load(f1)


fic_sav_DataBaseTest2 = "pickle/DataBaseTest2.pickle"
with open(fic_sav_DataBaseTest2, "rb") as DataBaseTest2:
    DataBaseTest = pickle.load(DataBaseTest2)


fic_sav_model_Forest1 = "pickle/model_Forest1.pickle"
with open(fic_sav_model_Forest1, "rb") as model_Forest1:
    model_Forest = pickle.load(model_Forest1)

# Pour obtenir l'index de l'ID
W = TestDash.index[TestDash.SK_ID_CURR == platform_name].values[0]
exp = Explainer.explain_instance(
    DataBaseTest.values[W], model_Forest.predict_proba, num_features=20
)

# exp = Explainer.explain_instance(data_row=Y.reshape(1, Y.shape[1])[0], predict_fn=model_Forest.predict_proba)
components.html(exp.as_html(), height=800)
exp.as_list()

st.markdown(
    "<h1 style='text-align: center; color: Black; font-size: 50px;'>ANALYSE GLOBALE ET POSITIONNEMENT DU CLIENT</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align: left; color: black; font-size: 30px;'>Age (ans) </h1>",
    unsafe_allow_html=True,
)

fig, ax = plt.subplots(figsize=(10, 6))
# KDE densite par noyau plot des prets qui ont ete remboursés
sns.kdeplot(TrainDash.loc[TrainDash["TARGET"] == 0, "DAYS_BIRTH"], label="target == 0")
# KDE plot des prets qui n'ont pas ete remboursés
sns.kdeplot(TrainDash.loc[TrainDash["TARGET"] == 1, "DAYS_BIRTH"], label="target == 1")
Z = TestDash[TestDash["SK_ID_CURR"] == platform_name]["DAYS_BIRTH"].item()
plt.axvline(x=Z, color="red", label="Position du client")

ZZ = TrainDash[TrainDash["TARGET"] == 0]["DAYS_BIRTH"].mean()
plt.axvline(x=ZZ, color="yellow", label='Moyenne d"âge TARGET=0')

ZZZ = TrainDash[TrainDash["TARGET"] == 1]["DAYS_BIRTH"].mean()
plt.axvline(x=ZZZ, color="blue", label='Moyenne d"âge TARGET=1')

# Labeling of plot
plt.xlabel("Age")
plt.ylabel("Density")
plt.legend()
plt.title("Distribution of Ages")
plt.show()
st.pyplot(fig)


st.markdown(
    "<h1 style='text-align: left; color: black; font-size: 30px;'> Ancienneté (ans) </h1>",
    unsafe_allow_html=True,
)

fig, ax = plt.subplots(figsize=(10, 6))
# KDE densite par noyau plot des prets qui ont ete remboursés
sns.kdeplot(TrainDash.loc[TrainDash["TARGET"] == 0, "DAYS_EMPLOYED"], label="target == 0")
# KDE plot des prets qui n'ont pas ete remboursés
sns.kdeplot(TrainDash.loc[TrainDash["TARGET"] == 1, "DAYS_EMPLOYED"], label="target == 1")
Z = TestDash[TestDash["SK_ID_CURR"] == platform_name]["DAYS_EMPLOYED"].item()
plt.axvline(x=Z, color="red", label="Position du client")

ZZ = TrainDash[TrainDash["TARGET"] == 0]["DAYS_EMPLOYED"].mean()
plt.axvline(x=ZZ, color="yellow", label="Ancienneté Moyenne TARGET=0")

ZZZ = TrainDash[TrainDash["TARGET"] == 1]["DAYS_EMPLOYED"].mean()
plt.axvline(x=ZZZ, color="blue", label="Ancienneté Moyenne TARGET=1")

# Labeling of plot
plt.xlabel("Ancienneté (en années)")
plt.ylabel("Density")
plt.legend()
plt.title("Distribution of DAYS_EMPLOYED")
plt.show()
st.pyplot(fig)


st.markdown(
    "<h1 style='text-align: left; color: black; font-size: 30px;'>Ext_Source_2 </h1>",
    unsafe_allow_html=True,
)

fig, ax = plt.subplots(figsize=(10, 6))
# KDE densite par noyau plot des prets qui ont ete remboursés
sns.kdeplot(TrainDash.loc[TrainDash["TARGET"] == 0, "EXT_SOURCE_2"], label="target == 0")
# KDE plot des prets qui n'ont pas ete remboursés
sns.kdeplot(TrainDash.loc[TrainDash["TARGET"] == 1, "EXT_SOURCE_2"], label="target == 1")
Z = TestDash[TestDash["SK_ID_CURR"] == platform_name]["EXT_SOURCE_2"].item()
plt.axvline(x=Z, color="red", label="Position du client")

ZZ = TrainDash[TrainDash["TARGET"] == 0]["EXT_SOURCE_2"].mean()
plt.axvline(x=ZZ, color="yellow", label="Moyenne Ext_Source_2 TARGET=0")

ZZZ = TrainDash[TrainDash["TARGET"] == 1]["EXT_SOURCE_2"].mean()
plt.axvline(x=ZZZ, color="blue", label="Moyenne Ext_Source_2 TARGET=1")

# Labeling of plot
plt.xlabel("Source Extérieure 2")
plt.ylabel("Density")
plt.legend()
plt.title("Distribution of EXT_SOURCE_2")
plt.show()
st.pyplot(fig)


st.markdown(
    "<h1 style='text-align: left; color: black; font-size: 30px;'>Ext_Source_3 </h1>",
    unsafe_allow_html=True,
)

fig, ax = plt.subplots(figsize=(10, 6))
# KDE densite par noyau plot des prets qui ont ete remboursés
sns.kdeplot(TrainDash.loc[TrainDash["TARGET"] == 0, "EXT_SOURCE_3"], label="target == 0")
# KDE plot des prets qui n'ont pas ete remboursés
sns.kdeplot(TrainDash.loc[TrainDash["TARGET"] == 1, "EXT_SOURCE_3"], label="target == 1")
Z = TestDash[TestDash["SK_ID_CURR"] == platform_name]["EXT_SOURCE_3"].item()
plt.axvline(x=Z, color="red", label="Position du client")

ZZ = TrainDash[TrainDash["TARGET"] == 0]["EXT_SOURCE_3"].mean()
plt.axvline(x=ZZ, color="yellow", label="Moyenne Ext_Source_3 TARGET=0")

ZZZ = TrainDash[TrainDash["TARGET"] == 1]["EXT_SOURCE_3"].mean()
plt.axvline(x=ZZZ, color="blue", label="Moyenne Ext_Source_3 TARGET=1")

# Labeling of plot
plt.xlabel("Source Extérieure 3")
plt.ylabel("Density")
plt.legend()
plt.title("Distribution of EXT_SOURCE_3")
plt.show()
st.pyplot(fig)


st.markdown(
    "<h1 style='text-align: left; color: black; font-size: 30px;'> Revenu Total ($) </h1>",
    unsafe_allow_html=True,
)

fig, ax = plt.subplots(figsize=(10, 6))
# KDE densite par noyau plot des prets qui ont ete remboursés
sns.kdeplot(TrainDash.loc[TrainDash["TARGET"] == 0, "AMT_INCOME_TOTAL"], label="target == 0")
# KDE plot des prets qui n'ont pas ete remboursés
sns.kdeplot(TrainDash.loc[TrainDash["TARGET"] == 1, "AMT_INCOME_TOTAL"], label="target == 1")
Z = TestDash[TestDash["SK_ID_CURR"] == platform_name]["AMT_INCOME_TOTAL"].item()
plt.axvline(x=Z, color="red", label="Position du client")

ZZ = TrainDash[TrainDash["TARGET"] == 0]["AMT_INCOME_TOTAL"].mean()
plt.axvline(x=ZZ, color="yellow", label="Moyenne du Revenu TARGET=0")

ZZZ = TrainDash[TrainDash["TARGET"] == 1]["AMT_INCOME_TOTAL"].mean()
plt.axvline(x=ZZZ, color="blue", label="Moyenne du Revenu TARGET=1")

# Labeling of plot
plt.xlabel("Revenu Total ($)")
plt.ylabel("Density")
plt.legend()
plt.title("Distribution of AMT_INCOME_TOTAL")
plt.show()
st.pyplot(fig)


#
st.markdown(
    "<h1 style='text-align: left; color: black; font-size: 30px;'> Revenu Total ($) </h1>",
    unsafe_allow_html=True,
)

fig, ax = plt.subplots(figsize=(10, 6))
# KDE densite par noyau plot des prets qui ont ete remboursés
sns.kdeplot(TrainDash.loc[TrainDash["TARGET"] == 0, "AMT_INCOME_TOTAL"], label="target == 0")
# KDE plot des prets qui n'ont pas ete remboursés
sns.kdeplot(TrainDash.loc[TrainDash["TARGET"] == 1, "AMT_INCOME_TOTAL"], label="target == 1")
Z = TestDash[TestDash["SK_ID_CURR"] == platform_name]["AMT_INCOME_TOTAL"].item()
plt.axvline(x=Z, color="red", label="Position du client")

ZZ = TrainDash[TrainDash["TARGET"] == 0]["AMT_INCOME_TOTAL"].mean()
plt.axvline(x=ZZ, color="yellow", label="Moyenne du Revenu TARGET=0")

ZZZ = TrainDash[TrainDash["TARGET"] == 1]["AMT_INCOME_TOTAL"].mean()
plt.axvline(x=ZZZ, color="blue", label="Moyenne du Revenu TARGET=1")

# Labeling of plot
plt.xlabel("Revenu Total ($)")
plt.ylabel("Density")
plt.legend()
plt.title("Distribution of AMT_INCOME_TOTAL")
plt.show()
st.pyplot(fig)


# x = TrainDash[TrainDash['TARGET']==0]['DAYS_BIRTH']
# y = TrainDash[TrainDash['TARGET']==1]['DAYS_BIRTH']
# z = TrainDash['DAYS
# _BIRTH']
