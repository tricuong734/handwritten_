"""Cac mo hinh thuoc NV2; moi cau hinh SVM duoc bao cao rieng."""
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def model_specs(seed=42):
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=1500, tol=1e-4),
            {"C": [0.1, 1.0, 10.0]},
        ),
        "knn": (
            KNeighborsClassifier(algorithm="brute", n_jobs=1),
            {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
        ),
        "svm_linear": (
            SVC(kernel="linear", cache_size=512),
            {"C": [0.1, 1.0]},
        ),
        "svm_rbf": (
            SVC(kernel="rbf", cache_size=512),
            {"C": [1.0, 5.0], "gamma": ["scale", 0.01]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=1),
            {"max_depth": [None, 25], "max_features": ["sqrt", 0.15]},
        ),
    }
