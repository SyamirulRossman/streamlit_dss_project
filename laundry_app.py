import streamlit as st
import numpy as np
import pandas as pd

from sklearn import preprocessing

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve

from matplotlib import pyplot as plt

st.title("DSS Project: Laundry decision support system")

df = pd.read_csv('LAUNDRY.csv')
df = df.drop(['No', 'Date', 'Time'], axis=1)

st.sidebar.subheader("Insert new data to predict the user basket size")

new_Pants_Colour = st.sidebar.text_input("Pants Colour", "black")
new_Age_Range = st.sidebar.number_input("Age Range", 0)
new_Basket_colour = st.sidebar.text_input("Basket Colour", "red")
new_Shirt_Colour = st.sidebar.text_input("Shirt Colour", "black")
new_Attire = st.sidebar.selectbox(
    "Attire", ("casual", "traditional", "formal", "unidentified"))
new_Body_Size = st.sidebar.selectbox("Body Size", ("thin", "moderate", "fat"))
new_Kids_Category = st.sidebar.selectbox("Kids Category",
                                         ("no_kids", "baby", "toddler", "young"))
new_Race = st.sidebar.selectbox(
    "Race", ("malay", "indian", "chinese", "foreigner", "unidentified"))
new_Washer_No = st.sidebar.number_input("Washer No", min_value=3, max_value=6)
new_Dryer_No = st.sidebar.number_input("Dryer No", min_value=7, max_value=10)


newData = pd.DataFrame({"Pants_Colour": [new_Pants_Colour], "Age_Range": [new_Age_Range], "Basket_colour": new_Basket_colour, "Shirt_Colour": new_Shirt_Colour,
                        "Attire": new_Attire, "Body_Size": new_Body_Size, "Kids_Category": new_Kids_Category, "Race": new_Race, "Washer_No": new_Washer_No, "Dryer_No": new_Dryer_No})


df_FS = df.copy()

st.header("Machine learning initiation")
st.write(" ")

if st.checkbox("View loaded dataset for machine learning:"):
    st.subheader("Raw loaded dataset")
    st.write(df_FS)

df_FS = df.drop(['With_Kids', 'shirt_type', 'pants_type',
                 'Wash_Item', 'Spectacles'], axis=1)

y = df_FS['Basket_Size']
df_FS = df_FS.drop(['Basket_Size'], axis=1)
y.fillna("small", inplace=True)


df_FS['Race'].fillna("unidentified", inplace=True)
df_FS['Body_Size'].fillna("moderate", inplace=True)
df_FS['Age_Range'].fillna(round(df_FS['Age_Range'].mean()), inplace=True)
df_FS['Kids_Category'].fillna("no_kids", inplace=True)
df_FS['Basket_colour'].fillna("red", inplace=True)
df_FS['Attire'].fillna("casual", inplace=True)
df_FS['Shirt_Colour'].fillna("black", inplace=True)
df_FS['Pants_Colour'].fillna("black", inplace=True)

if st.checkbox("View dataset that will be used for machine learning:"):
    st.subheader(
        "Processed dataset. (Only kept the data that useful for decision)")
    st.write(df_FS)

y.fillna("small", inplace=True)

df_FS = df_FS[['Pants_Colour', 'Age_Range', 'Basket_colour', 'Shirt_Colour',
               'Attire', 'Body_Size', 'Kids_Category', 'Race', 'Washer_No', 'Dryer_No']]

frames = [df_FS, newData]
df_FS = pd.concat(frames, ignore_index=True)
# st.write(df_FS)

label_encoder = preprocessing.LabelEncoder()
df_FS['Race'] = label_encoder.fit_transform(df_FS['Race'])
df_FS['Body_Size'] = label_encoder.fit_transform(df_FS['Body_Size'])
df_FS['Kids_Category'] = label_encoder.fit_transform(df_FS['Kids_Category'])
df_FS['Basket_colour'] = label_encoder.fit_transform(df_FS['Basket_colour'])
df_FS['Attire'] = label_encoder.fit_transform(df_FS['Attire'])
df_FS['Shirt_Colour'] = label_encoder.fit_transform(df_FS['Shirt_Colour'])
df_FS['Pants_Colour'] = label_encoder.fit_transform(df_FS['Pants_Colour'])

predict_set = df_FS.iloc[[-1]]

X = df_FS.iloc[:-1]
colnames = X.columns

st.subheader("Label encoded dataset")
st.write(X.head())

learning_name = st.selectbox(
    "Select Classifier/Clustering", ("Naive Bayes", "k-NN"))


def get_learning(ml_name):
    if ml_name == "Naive Bayes":
        ml = GaussianNB()
    else:
        ml = KNeighborsClassifier(n_neighbors=10)
    return ml


ml = get_learning(learning_name)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

ml.fit(X_train, y_train)
ml_pred = ml.predict(X_test)

st.write(f"Classifier = {learning_name}")

st.write(f"{learning_name} Score=", ml.score(X_test, y_test))
st.write('Precision= {:.2f}'.format(
    precision_score(y_test, ml_pred, pos_label="big")))
st.write('Recall= {:.2f}'. format(
    recall_score(y_test, ml_pred, pos_label="big")))
st.write('F1= {:.2f}'. format(
    f1_score(y_test, ml_pred, pos_label="big")))
st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, ml_pred)))

prob_ml = ml.predict_proba(X_test)
prob_ml = prob_ml[:, 1]

fpr_ml, tpr_ml, thresholds_ml = roc_curve(y_test, prob_ml, pos_label="big")

st.subheader("ROC Curve")

plt.plot(fpr_ml, tpr_ml, color='red', label=f'{learning_name}')
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()

st.pyplot(plt)

the_prediction = ml.predict(predict_set)

st.sidebar.subheader("Input data after Label Encoding:")
st.sidebar.write(predict_set)
st.sidebar.subheader("The result of prediction:")
st.sidebar.write("The basket size: " + the_prediction[0])
