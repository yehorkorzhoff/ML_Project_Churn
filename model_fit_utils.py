import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier


def add_performance_to_df(df_name, name_model, model, train_X, train_y, test_X, test_y):
    adder = {'model' : '', 'train_accuracy_score': '', 'train_roc_auc_score': '', 'train_proba_roc_auc_score': '',
             'train_mean_cross_val_score': '', 'test_accuracy_score': '',
             'test_roc_auc_score': '','test_proba_roc_auc_score': '', 'test_mean_cross_val_score': ''}
    
    train_predictions = model.predict(train_X)
    train_proba_predictions = model.predict_proba(train_X)[:, 1]
    test_predictions = model.predict(test_X)
    test_proba_predictions = model.predict_proba(test_X)[:, 1]
    
    adder['model'] = name_model
    adder['train_accuracy_score'] = accuracy_score(train_y, train_predictions)
    adder['test_accuracy_score'] = accuracy_score(test_y, test_predictions)
    adder['train_roc_auc_score'] = roc_auc_score(train_y, train_predictions)
    adder['train_proba_roc_auc_score'] = roc_auc_score(train_y, train_proba_predictions)
    adder['test_roc_auc_score'] = roc_auc_score(test_y, test_predictions)
    adder['test_proba_roc_auc_score'] = roc_auc_score(test_y, test_proba_predictions)
    adder['train_mean_cross_val_score'] = cross_val_score(model, train_X, train_y, cv=5).mean()
    adder['test_mean_cross_val_score'] = cross_val_score(model, test_X, test_y, cv=5).mean()
    
    
    return df_name.append(adder, ignore_index=True)


def get_poly_features(X, degree):
    poly = PolynomialFeatures(degree=degree)
    return poly.fit_transform(X)

def scale():
    pass


def fit_dt(X_train, y_train):
    dtc = DecisionTreeClassifier(random_state=17, class_weight='balanced')
    dtc.fit(X_train, y_train)
    d = {'DecisionTreeClassifier_train_test': dtc}
    return d


def fit_grid_search_dt(X_train, y_train, skf):
    dtc_params = {'criterion': ['entropy', 'gini'],
         'max_depth': range(1, 10),
         'min_samples_split': range(2, 8, 2),
         'min_samples_leaf': range(1, 5)}

    dtc_gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=17, class_weight='balanced'), param_grid=dtc_params,
                      scoring='roc_auc',
                cv=skf, n_jobs=-1, verbose=True)
    dtc_gs.fit(X_train, y_train)
    d = {'DecisionTreeClassifier_GridSearchCV': dtc_gs.best_estimator_}
    return d


def fit_lr(X_train, y_train):
    lr_train_test = LogisticRegression(random_state=17, class_weight='balanced')
    lr_train_test.fit(X_train, y_train)
    d = {'LogisticRegression_train_test': lr_train_test}
    return d


def fit_grid_search_lr(X_train, y_train, skf):
    lr_params = {'penalty': ['l1', 'l2', 'elasticnet'],
             'C': np.logspace(-4, 4, 20)}

    lr_gs = GridSearchCV(estimator=LogisticRegression(random_state=17, class_weight='balanced'), 
                      param_grid=lr_params, scoring='roc_auc',
                cv=skf, n_jobs=-1, verbose=True)
    lr_gs.fit(X_train, y_train)
    d = {'LogisticRegression_GridSearchCV': lr_gs.best_estimator_}
    return d


def fit_rf(X_train, y_train):
    rf_train_test = RandomForestClassifier(random_state=17, class_weight='balanced')
    rf_train_test.fit(X_train, y_train)
    d = {'RandomForestClassifier_train_test': rf_train_test}
    return d


def fit_grid_search_rf(X_train, y_train, skf):
    rf_params = {'n_estimators': range(25, 60, 5),
            'max_depth': list(range(10, 15, 5)), 
            'min_samples_leaf': list(range(7, 10)),
            'max_features': list(range(4,8, 2))}
    rf_gs = GridSearchCV(estimator= RandomForestClassifier(random_state=17, class_weight='balanced'), 
                      param_grid=rf_params, scoring='roc_auc',
                cv=skf, n_jobs=-1, verbose=True)
    rf_gs.fit(X_train, y_train)
    d = {'RandomForestClassifier_CV': rf_gs.best_estimator_}
    return d


def fit_knn(X_train, y_train):
    knn_train_test = KNeighborsClassifier()
    knn_train_test.fit(X_train, y_train)
    d = {'KNeighborsClassifier_train_test': knn_train_test}
    return d


def fit_grid_search_knn(X_train, y_train, skf):
    knn_params = {'n_neighbors': range(2, 10, 2),
              'weights': ['uniform', 'distance'],
               'p': [1, 2],
               'algorithm': ['ball_tree', 'kd_tree']}

    knn_gs = GridSearchCV(estimator= KNeighborsClassifier(), 
                      param_grid=knn_params, scoring='roc_auc',
                        cv=skf, n_jobs=-1, verbose=True)
    knn_gs.fit(X_train, y_train)
    d = {'KNeighborsClassifier_CV': knn_gs.best_estimator_}
    return d


def fit_bagg_model(model, X_train, y_train):
    model_params = list(model.values())[0].get_params()
    model_for_bag = type(list(model.values())[0])()
    bagg_clf = BaggingClassifier(model_for_bag, n_estimators=100, n_jobs=-1, random_state=17)
    bagg_clf.fit(X_train, y_train)
    d = {'BaggingClassifier' + list(model.keys())[0]: bagg_clf}
    return d


def get_models_performance(models, X_train, y_train, X_test, y_test):
    cols = ['model', 'train_accuracy_score', 'train_roc_auc_score', 'train_mean_cross_val_score', 'train_proba_roc_auc_score',
       'test_accuracy_score', 'test_roc_auc_score','test_proba_roc_auc_score', 'test_mean_cross_val_score']
    
    model_performance = pd.DataFrame(columns=cols)

    for key in models:
        model_performance = add_performance_to_df(model_performance, key, models[key],
                                                  X_train, y_train, X_test, y_test)
    return model_performance