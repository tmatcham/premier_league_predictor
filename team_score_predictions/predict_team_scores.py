import pandas as pd
import numpy as np
from data_downloaders.get_team_data import get_team_data, get_teams
from team_score_predictions.prepare_data import process_team_data_year
import scipy.stats as stats

train_years = [i for i in range(18,25)]

df_train = None
for year in train_years:
    df = get_team_data(year)
    df = process_team_data_year(df, 1)
    df_train = pd.concat([df_train, df])


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

#Let's make predictions for the current season based on the available data.
cur_year = 25
dfc = get_team_data(cur_year)
dfc = process_team_data_year(dfc, 1, True)

dfc = dfc[dfc['team_h_score'].isna()]
X = dfc.iloc[:,5:]
X = sm.add_constant(X)
team_h_pred_score = result1.predict(X)
team_a_pred_score = result2.predict(X)

dfo = dfc[['event', 'team_h', 'team_a']].copy()
dfo['team_h_pred_score'] = team_h_pred_score
dfo['team_a_pred_score'] = team_a_pred_score

teams = get_teams(25)
teams = teams[['id', 'short_name']]

dfo = pd.merge(dfo, teams, left_on='team_h', right_on='id')
dfo.drop(['id'], axis=1, inplace=True)
dfo.rename(columns={'short_name':'team_h_name'}, inplace=True)

dfo = pd.merge(dfo, teams, left_on='team_a', right_on='id')
dfo.drop(['id'], axis=1, inplace=True)
dfo.rename(columns={'short_name':'team_a_name'}, inplace=True)

dfo[dfo['event']==5][['team_h_name', 'team_a_name', 'team_h_pred_score', 'team_a_pred_score']].head()

dfo_h = dfo[['event', 'team_h', 'team_h_name', 'team_h_pred_score', 'team_a_pred_score']]
dfo_h.rename(columns={'team_h':'team', 'team_h_name':'name', 'team_h_pred_score': 'goals_scored', 'team_a_pred_score': 'goals_conceded'}, inplace=True)

dfo_a = dfo[['event', 'team_a', 'team_a_name', 'team_a_pred_score', 'team_h_pred_score']]
dfo_a.rename(columns={'team_a':'team', 'team_a_name':'name','team_a_pred_score': 'goals_scored', 'team_h_pred_score': 'goals_conceded'}, inplace=True)

df_out = pd.concat([dfo_h, dfo_a], axis = 0)


for i in np.unique(df_out['event']):
    temp = df_out[df_out['event']==i][['team', 'goals_scored', 'goals_conceded']]
    temp.rename(columns = {'goals_scored': 'week'+str(i)+'_pred_goals_scored', 'goals_conceded':'week'+str(i)+'_pred_goals_conceded'}, inplace = True)
    teams = pd.merge(teams, temp, left_on='id', right_on='team')
    teams.drop(['team'], axis=1, inplace=True)


teams.to_csv('data/team_score_predictions.csv', index=True)


