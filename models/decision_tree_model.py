import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def decision_tree(X, y):
    rmse_tree = []
    mae_tree = []
    r2_tree = []

    rmse_tree_train = []
    r2_tree_train = []

    param_grid_tree = {
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [10, 20, 30],
        'min_samples_leaf': [5, 10, 20]
    }

    for seed in range(20):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        base_tree = DecisionTreeRegressor(random_state=42)

        grid = GridSearchCV(
            base_tree,
            param_grid_tree,
            scoring='neg_root_mean_squared_error',
            cv=5
        )

        grid.fit(X_train, y_train)
        best_tree = grid.best_estimator_

        y_pred_test = best_tree.predict(X_test)
        y_pred_train = best_tree.predict(X_train)

        rmse_tree.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        mae_tree.append(mean_absolute_error(y_test, y_pred_test))
        r2_tree.append(r2_score(y_test, y_pred_test))

        rmse_tree_train.append(np.sqrt(mean_squared_error(y_train, y_pred_train)))
        r2_tree_train.append(r2_score(y_train, y_pred_train))

    return pd.DataFrame({
        'RMSE_teste': rmse_tree,
        'MAE_teste': mae_tree,
        'R2_teste': r2_tree,
        'RMSE_treino': rmse_tree_train,
        'R2_treino': r2_tree_train
    })

