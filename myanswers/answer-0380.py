import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score


def clasificar_resultado_con_mlp(X, y, capas_ocultas, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=capas_ocultas,
        max_iter=500,
        random_state=random_state,
    )
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
    }
