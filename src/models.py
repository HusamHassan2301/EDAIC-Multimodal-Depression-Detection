from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

SEED = 42

def get_models():
    return {
        "SVM": SVC(kernel="rbf", probability=True,
                   class_weight="balanced", random_state=SEED),
        "RF":  RandomForestClassifier(n_estimators=200,
                                      class_weight="balanced", random_state=SEED),
        "XGB": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                              eval_metric="logloss", random_state=SEED, verbosity=0),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64),
                             max_iter=400, random_state=SEED),
    }
