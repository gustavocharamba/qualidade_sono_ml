import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def RandomForestRegressionModel(X_train, X_test, y_train, y_test, X, y):
    rmse_rf = []
    mae_rf = []
    r2_rf = []

    rmse_rf_train = []
    r2_rf_train = []

    param_grid_rf = {
        'n_estimators': [100],
        'max_depth': [6, 10],
        'min_samples_leaf': [5, 10],
        'max_features': ['sqrt']
    }

    for seed in range(20):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)

        grid = GridSearchCV(
            base_rf,
            param_grid_rf,
            scoring='neg_root_mean_squared_error',
            cv=5,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)
        best_rf = grid.best_estimator_

        y_pred_test = best_rf.predict(X_test)
        y_pred_train = best_rf.predict(X_train)

        rmse_rf.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        mae_rf.append(mean_absolute_error(y_test, y_pred_test))
        r2_rf.append(r2_score(y_test, y_pred_test))

        rmse_rf_train.append(np.sqrt(mean_squared_error(y_train, y_pred_train)))
        r2_rf_train.append(r2_score(y_train, y_pred_train))

    return pd.DataFrame({
        'RMSE_teste': rmse_rf,
        'MAE_teste': mae_rf,
        'R2_teste': r2_rf,
        'RMSE_treino': rmse_rf_train,
        'R2_treino': r2_rf_train
    })