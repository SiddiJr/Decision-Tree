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

def fitting(X, y, criterion, splitting, mdepth, pred):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = tree.DecisionTreeClassifier(
                                        criterion=criterion,
                                        splitter=splitting,
                                        max_depth=mdepth,
                                        random_state=0
                                        )
    
    clf = model.fit(X_train, y_train)

    pred_label_train = model.predict(X_train)
    pred_label_test = model.predict(X_test)

    pred2 = model.predict(pred)
    

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
    print(classification_report(y_test, pred_label_test, zero_division=0))
    print('--------------------------------------------------------')
    print("")

    print('*************** Evaluation on Training Data ***************')
    score_tr = model.score(X_train, y_train)
    print('Accuracy Score: ', score_tr)
    print(classification_report(y_train, pred_label_train, zero_division=0))
    print('--------------------------------------------------------')


fitting(data[['qualPres', 'pulso', 'resp']], data['clas_grav'], 'entropy', 'best', mdepth=3, pred = df)