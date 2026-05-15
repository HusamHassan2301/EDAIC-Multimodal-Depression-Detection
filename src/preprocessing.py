import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

SEED = 42

def prepare(X_tr, y_tr, X_dv, X_te, k=100):
    """SelectKBest + StandardScaler + SMOTE. Fit on train only."""
    k = min(k, X_tr.shape[1])
    sel = SelectKBest(f_classif, k=k).fit(X_tr, y_tr)
    X_tr = sel.transform(X_tr)
    X_dv = sel.transform(X_dv)
    X_te = sel.transform(X_te)
    scaler = StandardScaler().fit(X_tr)
    X_tr   = scaler.transform(X_tr)
    X_dv   = scaler.transform(X_dv)
    X_te   = scaler.transform(X_te)
    X_tr_sm, y_sm = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)
    return X_tr_sm, y_sm, X_dv, X_te
