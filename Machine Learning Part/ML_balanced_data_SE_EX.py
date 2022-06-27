import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.calibration import CalibratedClassifierCV

from sklearn.decomposition import PCA

from sklearn.externals import joblib

# Folder to save models

try:
    os.mkdir("Models")
except Exception as e:
    print(e)

try:
    os.mkdir("Models/SE_EX")
except Exception as e:
    print(e)

# Features

df = pd.read_csv("featuressssssss_3/train_fetures_extraction_SE_EX.csv")
df_test = pd.read_csv("featuressssssss_3/test_fetures_extraction_SE_EX.csv")
print(df.head())
print(df_test.head())
print(df.shape)
print(df_test.shape)

df["name"] = df["name"].apply(lambda x: x.split("/")[-1])
df_test["name"] = df_test["name"].apply(lambda x: x.split("/")[-1])

# Making two new feautres of height and width


df["width"] = df["name"].apply(lambda x: x.split("_")[-3])
df["height"] = df["name"].apply(lambda x: x.split("_")[-4])
df_test["width"] = df_test["name"].apply(lambda x: x.split("_")[-3])
df_test["height"] = df_test["name"].apply(lambda x: x.split("_")[-4])

print(df.head())
print(df_test.head())

# Train Validation Split of Images

imgs = list(set(["_".join(i.split("_")[:2]) for i in df["name"].tolist()]))
print(imgs)
print(len(imgs))
train_ids = list(np.random.choice(imgs, int(len(imgs) * 0.8), replace = False))
valid_ids = [i for i in imgs if i not in train_ids]
print(train_ids)
print(valid_ids)

df_train = df[df["name"].apply(lambda x: True if "_".join(x.split("_")[:2]) in train_ids else False)]
df_valid = df[df["name"].apply(lambda x: True if "_".join(x.split("_")[:2]) in valid_ids else False)]
print(df_train.shape)
print(df_valid.shape)

print(df_train["Label"].value_counts())

# Balancing data

n_ma = df_train[df_train["Label"] == "SE"].shape[0]

df_he = df_train[df_train["Label"] == "SE"].sample(n_ma)
df_ma = df_train[df_train["Label"] == "EX"].sample(n_ma)
df_none = df_train[df_train["Label"] == "NONE"].sample(n_ma)

df_new = pd.concat((df_he, df_ma, df_none))

print(df_new.shape)
print(df_new["Label"].value_counts())


X_train = df_new.drop(["name", "Label", "cx", "cy"], axis = 1)
X_test = df_valid.drop(["name", "Label", "cx", "cy"], axis = 1)
X_TEST = df_test.drop(["name", "Label", "cx", "cy"], axis = 1)
y_train = df_new["Label"]
y_test = df_valid["Label"]
y_TEST = df_test["Label"]
print(X_train.shape)
print(y_train.shape)
X_train = X_train.values
y_train = y_train.values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, X_TEST.shape, y_TEST.shape)

# Scaling Features

ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
X_TEST = ss.transform(X_TEST)

scalar_filename = "Models/SE_EX/balanced_data_ss.save"
joblib.dump(ss, scalar_filename)

print("Logistic Regression")

model = LogisticRegression()
model.fit(X_train, y_train)
print(accuracy_score(y_train, model.predict(X_train)))
print(accuracy_score(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test)))
print(classification_report(y_TEST, model.predict(X_TEST)))

model_filename = "Models/SE_EX/balanced_data_lr.save"
joblib.dump(model, model_filename) 

print("Linear SVC")

model = LinearSVC()
model.fit(X_train, y_train)
print(accuracy_score(y_train, model.predict(X_train)))
print(accuracy_score(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test)))
print(classification_report(y_TEST, model.predict(X_TEST)))

print("SVC with RBF Kernel")

model = SVC(kernel = "rbf", probability = True, class_weight = "balanced")
model.fit(X_train, y_train)
print(accuracy_score(y_train, model.predict(X_train)))
print(accuracy_score(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test)))
print(classification_report(y_TEST, model.predict(X_TEST)))

model_filename = "Models/SE_EX/balanced_data_svc.save"
joblib.dump(model, model_filename) 

print("Random Forest")

model = RandomForestClassifier()
model.fit(X_train, y_train)
print(accuracy_score(y_train, model.predict(X_train)))
print(accuracy_score(y_test, model.predict(X_test)))
print(classification_report(y_test, model.predict(X_test)))
print(classification_report(y_TEST, model.predict(X_TEST)))

model_filename = "Models/SE_EX/balanced_data_rf.save"
joblib.dump(model, model_filename) 
