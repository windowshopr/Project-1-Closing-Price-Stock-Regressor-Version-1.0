import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFwe, SelectPercentile, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=9)

# Average CV score on the training set was:-5.215785947649995
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_regression, percentile=40),
    SelectFwe(score_func=f_regression, alpha=0.029),
    RandomForestRegressor(bootstrap=True, max_features=0.9000000000000001, min_samples_leaf=8, min_samples_split=15, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
