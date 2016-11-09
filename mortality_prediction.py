# -*- coding: utf-8 -*-

# import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt
# import seaborn as sns
# from datetime import datetime, timedelta
# import re
import drill_utilities as du
import warnings
import classif_utilities as cu
import xgboost as xgb
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn import base as base
import joblib

warnings.filterwarnings("ignore")

spark_home = "C:\spark152"
db_path = "D:/tmp/IPYNBv2/eds"
# spark_home = "/opt/mapr/spark/spark-1.6.1"
# db_path = "/user/mapr/eds"

class MortalityPredML:

    def __init__(self):
        # Connect to Drill database
        self._dq = du.DrillODBC("drill_eds")
        # Compute APHP probability threshold
        logit = -7.7631 + 0.0737 * 15. + 0.9971 * np.log(16.)
        self._aphp_thres = np.exp(logit) / (1. + np.exp(logit))
        # Variable of interest to keep
        self._vois = ['int_esc_admission', 'int_esc_age', 'int_esc_bicarbonate', 'int_esc_bilirubine',
                      'int_esc_diurese', 'int_esc_frequence', 'int_esc_glasgow', 'int_esc_globules',
                      'int_esc_kaliemie', 'int_esc_natremie', 'int_esc_pao2', 'int_esc_pression',
                      'int_esc_temperature', 'int_esc_uree', 'int_maladies_chronique']
        # Mapping between form vois codes and Orbis reference catalog
        self._voi_map = {
            'int_esc_admission': 'TYPEADMISSION',
            'int_esc_age': 'AGE',
            'int_esc_bicarbonate': 'BICARBONATEMIE',
            'int_esc_bilirubine': 'BILIRUBINE',
            'int_esc_diurese': 'DIURESE',
            'int_esc_frequence': 'FREQUENCECARDIAQUE',
            'int_esc_glasgow': 'GLASGOW',
            'int_esc_globules': 'GLOBULESBLANCS',
            'int_esc_kaliemie': 'KALIEMIE',
            'int_esc_natremie': 'NATREMIE',
            'int_esc_pao2': 'PAO2/FIO2',
            'int_esc_pression': 'PRESSIONARTERIELLE',
            'int_esc_temperature': 'TEMPERATURE',
            'int_esc_uree': 'UREE',
            'int_maladies_chronique': 'MALADIESCHRONIQUES'
        }
        self._df_categories = None
        self.load_table()
        self.load_categories()
        self.valid_indices = None

    def __len__(self):
        return 1

    def load_table(self):
        # Load dataset from Drill database
        self._table = self._dq.df_from_query("select * from igs2_dataset")
        # Target vector
        self._y_true = np.logical_not((self._table['dt_deces'].isnull()) | (self._table['dt_deces'] == 'None')).values
        # Matrix of examples
        self._X = self._table[self._vois]
        # rename columns
        self._X.rename(columns=self._voi_map, inplace=True)
        self._dataset = self._X.copy()

    def load_categories(self):
        q = "select PARENT as SAPS2_VARIABLE, LID as SAPS2_INTERVAL, saps2weight, index from \
            IGS2_ref_feature_categories order by PARENT, index"
        self._df_categories = self._dq.df_from_query(q)
        self._df_categories.rename(columns={'PARENT': 'SAPS2_VARIABLE', 'LID': 'SAPS2_INTERVAL'}, inplace=True)

    def init_X(self):
        self._X = self._dataset.copy()

    def label_encode(self):
        # Create label encoder for each voi
        label_encoders = {}
        for key in self._voi_map.keys():
            voi = self._voi_map[key]
            # DataFrame restriction
            df_labels = self._df_categories[self._df_categories['SAPS2_VARIABLE'] == voi]
            le = LabelEncoder()
            list_labels = df_labels['saps2weight'].values.astype('str')
            le.fit(list_labels)
            label_encoders[voi] = le
            self._X[voi] = le.transform(self._X[voi].astype('str'))

    def add_features(self, l_var):
        # between 0 and 6 variable to derive
        nb_var = 0
        if l_var is not None:
            nb_var = len(l_var)
        if nb_var > 0:
            for f in l_var:
                cpt_name = 'C_' + f
                self._X[cpt_name] = self._X.groupby(f)[f].transform('count')
                # sq_name = 'SQ_' + f
                # self._X[sq_name] = np.sqrt(self._X[f].astype(float))
                ln_name = 'LN_' + f
                self._X[ln_name] = np.log(self._X[f].astype(float) + 1.)
                # mean = self._X[f].astype(float).mean()
                # mean_name = 'DMEAN_' + f
                # self._X[mean_name] = np.abs(self._X[f].astype(float) - mean)
        return l_var

    def prepare_sets(self):
        # Separate train / validation
        if self.valid_indices == None:
            self.X_train, self.y_train, self.X_valid, self.y_valid, self.valid_indices = \
                cu.split_dataset(self._X.values, self._y_true, indices=True)
            # Retrieve related validation set from the resultf DataFrame
            self.resultf_valid = self._table.ix[self.valid_indices, :]
            self.yf_pred = self.resultf_valid['pr'] > self._aphp_thres
            self.train_indices = np.setdiff1d(np.arange(len(self._X)), self.valid_indices)
        else:
            # keep always the same validation set (to be able to compare with classical SAPS2 score)
            self.X_valid = self._X.values[self.valid_indices]
            self.y_valid = self._y_true[self.valid_indices]
            self.X_train = self._X.values[self.train_indices]
            self.y_train = self._y_true[self.train_indices]

    @staticmethod
    def get_mortality_rate(y):
        return float(np.size(np.where(y == True)[0])) / float(np.size(np.where(y == False)[0])) * 100.

    @staticmethod
    def get_scores(y_pred, y_true, y_proba):
        return accuracy_score(y_true, y_pred), recall_score(y_true, y_pred), \
               precision_score(y_true, y_pred), roc_auc_score(y_true, y_proba)

    def get_saps2_auc(self):
        pred = self._table.loc[self.valid_indices]['pr']
        return roc_auc_score(self.y_valid, pred)


def gen_list_of_add_features(max_eval, space):
    m_pred = space['MortalityPredMLObject']
    columns = m_pred._voi_map.values()
    for i in range(max_eval):
        nb_var = np.random.randint(6)
        l_var = None
        if nb_var > 0:
            l_var = list(np.random.choice(np.array(columns), nb_var))
        list_of_add_features.append(l_var)
    space['add_features'] = hp.choice('x_add_features', list_of_add_features)


def objective(space_params):
    clf = xgb.XGBClassifier(n_estimators=int(space_params['n_estimators']),
                            max_depth=int(space_params['max_depth']),
                            min_child_weight=space_params['min_child_weight'],
                            subsample=space_params['subsample'],
                            colsample_bytree=space_params['colsample_bytree']
                            #####
                            ,
                            colsample_bylevel=space_params['colsample_bylevel'],
                            base_score=space_params['base_score'],
                            scale_pos_weight=space_params['scale_pos_weight']
                            #####
                            ,
                            objective='binary:logistic'
                            )
    global best_clf
    global max_score
    # retrieve MortalityPredML object in the space
    m_predict = space_params['MortalityPredMLObject']
    # initialize dataset
    m_predict.init_X()
    if space_params['label_encode']:
        m_predict.label_encode()
    # Add features randomly
    l_added_vars = m_predict.add_features(space_params['add_features'])
    # split train and validation sets
    m_predict.prepare_sets()
    # ????
    eval_set = [(m_predict.X_train, m_predict.y_train), (m_predict.X_valid, m_predict.y_valid)]
    # fit classifier
    clf.fit(m_predict.X_train, m_predict.y_train, eval_set=eval_set, eval_metric="auc", \
            early_stopping_rounds=30)
    # get predict probabilities to compute the AUC
    pred = clf.predict_proba(m_predict.X_valid)[:, 1]
    # compute related AUC
    auc = roc_auc_score(m_predict.y_valid, pred)
    clf_inter = None
    if auc > max_score:
        max_score = auc
        best_clf = base.clone(clf)
        clf_inter = clf
    print("SCORE:", auc, "ADDED FEATURES:", l_added_vars)
    space_params['added_features'] = l_added_vars
    if clf_inter is not None:
        clf = best_clf
        best_clf = clf_inter
    return {'loss': 1 - auc, 'status': STATUS_OK}


def test_clf(filename, m_predict, l_added_feats, label_encode=False):
    clf = joblib.load(filename)
    # initialize dataset
    m_predict.init_X()
    if label_encode:
        m_predict.label_encode()
    # Add features randomly
    l_added_vars = m_predict.add_features(l_added_feats)
    # split train and validation sets
    m_predict.prepare_sets()
    # get predict probabilities to compute the AUC
    pred = clf.predict_proba(m_predict.X_valid)[:, 1]
    # print(clf.predict_proba(m_predict.X_valid))
    # compute related AUC
    auc = roc_auc_score(m_predict.y_valid, pred)
    print("SCORE:", auc, "ADDED FEATURES:", l_added_vars)

__name__ = '__main1__'

if __name__ == '__main1__':
    global max_score
    global best_clf
    best_clf = None
    max_score = 0.
    max_eval = 300
    # Define space for Hyperopt
    # space = {
    #     'label_encode': hp.choice('x_label_encode', [False]),
    #     'n_estimators': hp.quniform("x_n_estimators", 40, 200, 1),
    #     'max_depth': hp.quniform("x_max_depth", 2, 15, 1),
    #     'min_child_weight': hp.uniform('x_min_child', 1, 10),
    #     'subsample': hp.uniform('x_subsample', 0.7, 1),
    #     #####
    #     'base_score': hp.uniform('x_base_score', 0.3, 0.8),
    #     'scale_pos_weight': hp.uniform('x_scale_pos_weight', 0.7, 1),
    #     'colsample_bylevel': hp.uniform('x_colsample_bylevel', 0.7, 1),
    #     #####
    #     'colsample_bytree': hp.uniform('x_colsample_bytree', 0.7, 1),
    #     'MortalityPredMLObject': MortalityPredML()
    # }
    space = {
        'label_encode': hp.choice('x_label_encode', [False]),
        'n_estimators': hp.quniform("x_n_estimators", 40, 45, 1),
        'max_depth': hp.quniform("x_max_depth", 7, 8, 1),
        'min_child_weight': hp.uniform('x_min_child', 2.5, 3),
        'subsample': hp.uniform('x_subsample', 0.7, 0.75),
        #####
        'base_score': hp.uniform('x_base_score', 0.3, 0.8),
        'scale_pos_weight': hp.uniform('x_scale_pos_weight', 0.85, 0.9),
        'colsample_bylevel': hp.uniform('x_colsample_bylevel', 0.75, 0.8),
        #####
        'colsample_bytree': hp.uniform('x_colsample_bytree', 0.88, 0.9),
        'MortalityPredMLObject': MortalityPredML()
    }


    trials = Trials()
    # list_of_add_features = []
    # gen_list_of_add_features(max_eval, space)
    list_of_add_features = [['FREQUENCECARDIAQUE', 'BILIRUBINE']] * max_eval
    space['add_features'] = hp.choice('x_add_features', list_of_add_features)
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_eval,
                trials=trials)

    print("BEST ==>")
    print(best)
    print(list_of_add_features[best['x_add_features']])
    print(max_score)
    print("saps2 AUC={}".format(space['MortalityPredMLObject'].get_saps2_auc()))
    joblib.dump(best_clf, "best_saps2_xgb.pkl")


if __name__ == '__main2__':
    # params = {
    #     'max_depth': 4,
    #     'min_child_weight': 5.5308645266021905,
    #     'subsample': 0.8392713386393921,
    #     'scale_pos_weight': 0.8338763461794458,
    #     'colsample_bylevel': 0.9061283838410611,
    #     'colsample_bytree': 0.7767847025733484,
    #     'base_score': 0.691094181130932,
    #     'n_estimators': 69}
    added_feat = ['TYPEADMISSION', 'NATREMIE', 'FREQUENCECARDIAQUE', 'TEMPERATURE']
    m_predict = MortalityPredML()
    test_clf("best_saps2_xgb.pkl", m_predict, added_feat)