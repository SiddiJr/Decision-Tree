import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import graphviz

col_names = ['ID', 'p_Sist', 'p_Diast','qualPres', 'pulso', 'resp', 'gravidade', 'clas_grav']
data = pd.read_csv('datasets/data_20x20_42vic/sinais_vitais.txt', names=col_names, index_col = None)
data = data[['qualPres', 'pulso', 'resp', 'clas_grav']]
df = pd.read_csv('datasets/data_800vic/sinais_vitais_sem_label.txt', names=col_names, index_col = None)
df = df[['qualPres', 'pulso', 'resp']]
y = pd.read_csv('datasets/data_800vic/sinais_vitais_com_label.txt', names=col_names, index_col = None)
y = y[['clas_grav']]

def fitting(X, y, criterion, splitting, mdepth, X_pred, y_pred):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = tree.DecisionTreeClassifier(
                                        criterion=criterion,
                                        splitter=splitting,
                                        max_depth=mdepth,
                                        max_leaf_nodes=6,
                                        random_state=0
                                        )
    
    clf = model.fit(X_train, y_train)

    pred_label_train = model.predict(X_train)
    pred_label_test = model.predict(X_pred)
    

    print('*************** Tree Summary ***************')
    print('Classes: ', clf.classes_)
    print('Tree Depth: ', clf.get_depth())
    print('No. of leaves: ', clf.get_n_leaves())
    print('No. of features: ', clf.n_features_in_)
    print('--------------------------------------------------------')
    print("")
    
    print('*************** Evaluation on Test Data ***************')

    score_te = model.score(X_test, y_test)
    print('Accuracy Score: ', score_te)
    print(classification_report(y_pred, pred_label_test, zero_division=0))
    print('--------------------------------------------------------')
    print("")

    print('*************** Evaluation on Training Data ***************')
    score_tr = model.score(X_train, y_train)
    print('Accuracy Score: ', score_tr)
    print(classification_report(y_train, pred_label_train, zero_division=0))
    print('--------------------------------------------------------')


fitting(data[['qualPres', 'pulso', 'resp']], data['clas_grav'], 'entropy', 'random', mdepth=4, X_pred = df, y_pred = y)