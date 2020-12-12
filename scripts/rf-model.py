### Adapted from https://github.com/carlosbkm/car-destination-prediction/blob/master/random-forest-model.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geohash
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from haversine import haversine, Unit

# Get the longitude and latitude from the geohash
def decodegeo(geo, which):
    if len(geo) >= 6:
        geodecoded = geohash.decode(geo)
        return geodecoded[which]
    else:
        return 0


def further_data_prep(df):
    df['start_lat'] = df['location_start'].apply(lambda geo: decodegeo(geo, 0))
    df['start_lon'] = df['location_start'].apply(lambda geo: decodegeo(geo, 1))
    df['end_lat'] = df['location_end'].apply(lambda geo: decodegeo(geo, 0))
    df['end_lon'] = df['location_end'].apply(lambda geo: decodegeo(geo, 1))

    return df

# Funtion for cross-validation over a grid of parameters

def cv_optimize(clf, parameters, X, y, n_jobs=1, n_folds=5, score_func=None):
    if score_func:
        gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds, n_jobs=n_jobs, scoring=score_func)
    else:
        gs = GridSearchCV(clf, param_grid=parameters, n_jobs=n_jobs, cv=n_folds)
    gs.fit(X, y)
    print ("BEST", gs.best_params_, gs.best_score_, gs.cv_results_)
    best = gs.best_estimator_
    return best

def model(featuredDataset):
    columns_all_features = featuredDataset.columns
    columns_X = ['day_num', 'x_start', 'y_start', 'z_start']
    columns_y = ['end_lat', 'end_lon']
    X = featuredDataset[columns_X]
    y = featuredDataset[columns_y]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('X: ({}, {})'.format(*X.shape))
    print('y: ({}, {})'.format(*y.shape))
    print('X_train: ({}, {})'.format(*X_train.shape))
    print('y_train: ({}, {})'.format(*y_train.shape))
    print('X_test: ({}, {})'.format(*X_test.shape))
    print('y_test: ({}, {})'.format(*y_test.shape))

    # Set the parameters by cross-validation
    tuned_parameters = {'n_estimators': [2, 5, 10, 20, 40], 'max_depth': [None, 1, 2, 3, 4],
                        'min_samples_split': [2, 3, 4, 5, 6]}

    # clf = ensemble.RandomForestRegressor(n_estimators=500, n_jobs=1, verbose=1)
    gridCV = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=2, n_jobs=-1, verbose=1)
    gridCV.fit(X_train, y_train)
    print(gridCV.best_estimator_)

    reg = gridCV.best_estimator_
    training_accuracy = reg.score(X_train, y_train)
    valid_accuracy = reg.score(X_test, y_test)
    rmsetrain = np.sqrt(mean_squared_error(reg.predict(X_train), y_train))
    rmsevalid = np.sqrt(mean_squared_error(reg.predict(X_test), y_test))
    print(" R^2 (train) = %0.6f, R^2 (valid) = %0.6f, RMSE (train) = %0.6f, RMSE (valid) = %0.6f" % (training_accuracy, valid_accuracy, rmsetrain, rmsevalid))
    return reg, X, X_train

if __name__ == '__main__':
    os.chdir('..')
    df = pd.read_csv("./data/featured-dataset.csv")
    df = df.drop(df.columns[0], axis=1)
    featuredDataset = further_data_prep(df)
    reg, X, X_train = model(featuredDataset)
    importances = reg.feature_importances_
    std = np.std([tree.feature_importances_ for tree in reg.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    feature_names = X_train.columns
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), feature_names)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

