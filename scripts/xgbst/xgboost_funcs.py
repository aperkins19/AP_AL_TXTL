import xgboost
from sklearn.metrics import mean_absolute_error

def xgboost_define_model(params):
    """Defines regressor model and returns it"""
    model = xgboost.XGBRegressor(
                        max_depth = params["max_depth"],
                        min_child_weight = params["min_child_weight"],
                        learning_rate = params["learning_rate"],
                        objective= params["objective"],
                        early_stopping_rounds=10,
                        seed=578)
    return model

def xgboost_fit(model, X_train, y_train, X_test, y_test):
    """"Fits model using split data and returns metrics"""


    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
                )

    eval_preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, eval_preds)

    return mae