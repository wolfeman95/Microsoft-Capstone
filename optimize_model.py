import itertools
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

data_set = pd.read_csv("capstone_refine.csv")
training_labels = data_set[["poverty_rate"]]
training_set = data_set.drop(["poverty_rate"], axis=1)

def opt_params():
    """Optimizes parameters of the GradientBoostingRegressor"""
    x_train, x_test, y_train, y_test = train_test_split(training_set, training_labels, train_size=0.85,
                                                        test_size=0.15, random_state=7)
    lowest_rmse = 0
    first_loop = True
    best_params = []
    for est in range(50, 100, 5):
        for samp in range(30, 50, 2):
            for depth in range(25, 35, 1):
                regr = GradientBoostingRegressor(learning_rate=0.109, n_estimators=est,
                                                 min_samples_leaf=samp, max_depth=depth)
                regr.fit(x_train, y_train.values.ravel())
                y_2 = regr.predict(x_test)
                rmse = metrics.mean_squared_error(y_test, y_2)
                if first_loop:
                    lowest_rmse += rmse
                    first_loop = False
                else:
                    if rmse < lowest_rmse:
                        del best_params[:]
                        best_params += [est, samp, depth]
                        lowest_rmse = 0
                        lowest_rmse += rmse
                print("num est = ", est, "min samp per leaf = ", samp, "max depth = ", depth)
                print("RMSEe = ", rmse)
                print("Lowest RMSE = ", lowest_rmse)
                if best_params:
                    print("best params: num est = ", best_params[0], "min samp per leaf = ",
                          best_params[1], "max depth = ", best_params[2])
                print("-----------------------------------------------------")


def optimize_feature_space(potential_dropped_features):
    """Discovers which combination of dropped features makes our model better"""
    lowest_rmse = 0
    first_loop = True
    dropped_features_that_improved_model = []
    for combo in range(0, len(potential_dropped_features)+1):
        for subset in itertools.combinations(potential_dropped_features, combo):
            x_train, x_test, y_train, y_test = train_test_split(training_set.drop(list(subset), axis=1),
                                                                training_labels, train_size=0.85,
                                                                test_size=0.15, random_state=7)

            regr = GradientBoostingRegressor(n_estimators=754, learning_rate=0.109,
                                             min_samples_leaf=35, max_depth=28)
            regr.fit(x_train, y_train.values.ravel())
            y_2 = regr.predict(x_test)
            rmse = metrics.mean_squared_error(y_test, y_2)
            if first_loop:
                lowest_rmse += rmse
                first_loop = False
            else:
                if rmse < lowest_rmse:
                    lowest_rmse = 0
                    del dropped_features_that_improved_model[:]
                    lowest_rmse += rmse
                    dropped_features_that_improved_model += list(subset)
            print("FEATURES DROPPED: ", subset)
            print("RMSEe = ", rmse)
            print("Lowest RMSE = ", lowest_rmse)
            print("Best Dropped: ", dropped_features_that_improved_model)
            print("--------------------------------------------------------")
