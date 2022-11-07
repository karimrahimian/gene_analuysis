from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score, roc_auc_score
from sklearn.model_selection import  KFold, train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import glob
def ReplaceNan(df):
    newdf=df
    numerical_columns = newdf._get_numeric_data().columns
    for column in numerical_columns:
        newdf[column].fillna(value=newdf[column].mean(), inplace=True)
    return newdf
def One_Hot_Encoding(df):
    newdf=df

    numerical_columns = newdf._get_numeric_data().columns
    skipped_column = {'Sample ID'}
    categorical_columns = set(newdf.columns).difference(set(numerical_columns)).difference(skipped_column)
    #categorical_columns = ['VirusStatus(Hepatit C)', 'VirusStatus (Hepatit B)' , 'gender']
    for column in categorical_columns:
        one_hot = pd.get_dummies(newdf[column])
        newdf = newdf.drop(column, axis=1)
        newdf = newdf.join(one_hot,lsuffix='_left')
    return newdf
def Drop_Columns(df):
    newdf = df
    dropped_columns = ['sample','Corected Stage']
    for column in dropped_columns:
        newdf = newdf.drop(column,axis=1)
    return newdf

def One_Label_Encoding(df):
    newdf=df
    columns = ['DRB1_allele1','DRB1_allele2','DQB1_allele1','A_allele1','A_allele2','B_allele1','B_allele2']
    skipped_column = {'Sample ID','Corected Stage'}
    numerical_columns = newdf._get_numeric_data().columns
    categorical_columns = set(newdf.columns).difference(set(numerical_columns)).difference(skipped_column)
    for column in categorical_columns:
        categorical = newdf[column].astype('category')
        newdf[column] = categorical.cat.codes
    return newdf
def PreProcessExcelFile(filename):

    dataframe = pd.read_excel(io=filename)
    #dataframe = One_Hot_Encoding(dataframe)

    dataframe = One_Label_Encoding(dataframe)
    dataframe = ReplaceNan(dataframe)
    dataframe = Drop_Columns(dataframe)
    fn = os.path.basename(filename).split('/')[-1]
    dataframe.to_excel("preprocessed/preprocessed_{}.xlsx".format(fn[:-5]))
    return dataframe
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)
def RemoveCorrelated( dataframe):
    corr_matrix = dataframe.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = {}
    for i in range(upper.values.shape[0]):
        for j in range(i + 1, upper.values.shape[0]):
            if upper.values[i, j] >= 0.70:
                to_drop[upper.columns[j]] = 1

    uncorrelated_data = dataframe.drop(to_drop.keys(), axis=1)
    return uncorrelated_data
def doClassifyCrossValidation(X, y, classifier, nfold=5):
    evaluationName = ['Precision', 'F1Score', 'Accuracy', 'Recall', 'Matt', 'Auc']
    Data = X
    evlP = [[0 for x in range(6)] for YY in range(nfold)]
    k = 0
    kf = KFold(n_splits=nfold, shuffle=False)
    for train_index, test_index in kf.split(Data):
        classifier.fit(Data[train_index], y[train_index])
        y_pred = classifier.predict(Data[test_index])
        y_test = y[test_index]

        evlP[k][0] = (precision_score(y_test, y_pred, average='micro'))
        evlP[k][1] = (f1_score(y_test, y_pred, average='macro'))
        evlP[k][2] = (accuracy_score(y_test, y_pred))
        evlP[k][3] = (recall_score(y_test, y_pred, average="weighted"))
        evlP[k][4] = (matthews_corrcoef(y_test, y_pred))
        evlP[k][5] = multiclass_roc_auc_score(y_test, y_pred)
        print(evlP[k])
        k += 1

    average = np.matrix(evlP)
    average = average.mean(axis=0)
    average = np.squeeze(np.asarray(average))
    modelparams = pd.DataFrame({'Evaluating Function': evaluationName, 'Values': average})
    return modelparams
def Normalize (X):
    X = RobustScaler().fit_transform(X)
    return X
def SplitFeatureFromTarget(dataframe):
    df = dataframe
    target_column = 'Os. Time2'
    target = df[target_column]
    sampleID = df['Sample ID']
    df = df.drop(target_column, axis=1)
    df = df.drop('Sample ID', axis=1)

    X = np.array(df.to_numpy())
    X = X.astype(np.float)
    y = target.to_numpy()
    return X,y,sampleID

dataframe = PreProcessExcelFile('data/xlsx/data.xlsx')
#uncorrelated = RemoveCorrelated(dataframe)
X,y,sampleID = SplitFeatureFromTarget(dataframe)
X = Normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

linReg = LinearRegression().fit(X_train,y_train)
yhat = linReg.predict(X_test)
mae = metrics.mean_absolute_error(y_test, yhat)
mse = metrics.mean_squared_error(y_test, yhat)
rmse = np.sqrt(mse) # or mse**(0.5)
r2 = metrics.r2_score(y_test,yhat)
print(r2)




