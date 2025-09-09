import pandas as pd
import numpy as np
from data_downloaders import get_team_data
import matplotlib.pyplot as plt
from team_score_predictions.prepare_data import process_team_data_year
import scipy.stats as stats

train_years = [i for i in range(18,23)]
test_years = [23,24]

df_train = None
for year in train_years:
    df = get_team_data.get_team_data(year)
    df = process_team_data_year(df)
    df_train = pd.concat([df_train, df])
# want to prepare the data

df_test = None
for year in test_years:
    df = get_team_data.get_team_data(year)
    df = process_team_data_year(df)
    df_test = pd.concat([df_test, df])

import statsmodels.api as sm
# Now create a Poisson regression model that can be used to make future predictions.
# Remove incomplete rows
y1_train = df_train.pop('team_h_score')
y2_train = df_train.pop('team_a_score')
X_train = df_train
X_train = sm.add_constant(X_train)

model1 = sm.GLM(
    y1_train,
    X_train,
    family=sm.families.Poisson(),
)
result1 = model1.fit()

model2 = sm.GLM(
    y2_train,
    X_train,
    family=sm.families.Poisson(),
)
result2 = model2.fit()

# Now make predictions for held out data
y1_test = df_test.pop('team_h_score')
y2_test = df_test.pop('team_a_score')
X_test = df_test
X_test = sm.add_constant(X_test)


# Some model inspection
models = [result1, result2]
ys = [y1_test, y2_test]
for i in range(2):
    predictions = list(models[i].predict(X_test))
    #Find the indices for predictions in different ranges
    fig = plt.figure()
    for k,l in enumerate([0.5, 1, 1.5, 2, 2.5]):
        ind = [j for j in range(len(predictions)) if (predictions[j] > l) and predictions[j] < (l+0.5)]
        y1s_ind = [ys[i].iloc[l] for l in ind]
        ll = l + 0.25
        pmf = [len(y1s_ind)*stats.poisson.pmf(k ,mu = ll) for k in range(10)]

        plt.subplot(3, 2, k+1)
        plt.hist(y1s_ind)
        plt.plot(range(10), pmf)


# Fit looks good.

#Let's make predictions for the current season based on the available data.