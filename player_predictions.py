

import requests, json
import pandas as pd
import numpy as np
from pprint import pprint

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

base_url = 'https://fantasy.premierleague.com/api/'

# get data from bootstrap-static endpoint
r = requests.get(base_url+'bootstrap-static/').json()

# show the top level fields
pprint(r, indent=2, depth=1, compact=True)

# get player data from 'elements' field
players = r['elements']
players = pd.json_normalize(r['elements'])

#players = players.drop(71)
#Go by team
teams = pd.read_csv('data/team_score_predictions.csv', index_col=0)

players = pd.merge(players, teams[['id', 'short_name']], left_on = 'team', right_on = 'id')
players = players.rename(columns={'id_x':'id', 'short_name':'team_name'})
players = players.drop(['id_y'], axis = 1)
# How many weeks in the future should the score prediction be for?


players['mins3'] = 0
players['g5'] = 0
players['a5'] = 0
players['xg5'] = 0
players['xa5'] = 0
player_ids = players['id']
for i,id in enumerate(players['id']):
    print(str(i)+' out of '+ str(len(player_ids)))
    r = requests.get(base_url+'element-summary/'+str(id)).json()

    #if the history is empty then remove the player from players
    if r['history'] == []:
        players = players[players['id']!=id]
    else:
        mins = pd.DataFrame(r['history'])['minutes'].rolling(3).mean().tail(1)
        g5 = pd.DataFrame(r['history'])['goals_scored'].rolling(4).sum().tail(1)
        xg5 = pd.DataFrame(r['history'])['expected_goals'].rolling(4).sum().tail(1)
        a5 = pd.DataFrame(r['history'])['assists'].rolling(4).sum().tail(1)
        xa5 = pd.DataFrame(r['history'])['expected_assists'].rolling(4).sum().tail(1)

        players.loc[players['id']==id, 'mins3'] = mins.values
        players.loc[players['id']==id, 'g5'] = g5.values
        players.loc[players['id']==id, 'a5'] = a5.values
        players.loc[players['id']==id, 'xg5'] = xg5.values
        players.loc[players['id']==id, 'xa5'] = xa5.values

#save player data so we dont need to make requests again.
players.to_csv('data/player_data.csv', index=True)
players = pd.read_csv('data/player_data.csv', index_col=0)


weeks_ahead = 3
current_week = 4
players2 = None

for id in teams['id']:
    #get all the players
    team_players = players.loc[players['team']==id]
    team = teams.loc[teams['id'] == id]

    team_players['expected_goals_fraction'] = team_players['g5']/np.sum(team_players['g5'])
    team_players['expected_assists_fraction']  = team_players['xa5']/np.sum(team_players['xa5'])

    for week in range(weeks_ahead):
        expected_goals = np.asarray(teams.loc[teams['id']==id, 'week'+str(current_week+week+1)+'_pred_goals_scored'])[0]
        clean_sheet_prob = np.asarray(np.exp(-teams.loc[teams['id']==id, 'week'+str(current_week+week+1)+'_pred_goals_conceded']))[0]

        team_players['week'+str(current_week+week+1)+'_score_exp'] = 0

        #add to score for minutes played
        team_players.loc[team_players['mins3']>= 60, 'week'+str(current_week+week+1)+'_score_exp'] += 2
        team_players.loc[(team_players['mins3']> 0) * (team_players['mins3']<60), 'week'+str(current_week+week+1)+'_score_exp'] += 1

        #add to score for goals
        team_players.loc[team_players['element_type']==4, 'week'+str(current_week+week+1)+'_score_exp'] += 4* team_players.loc[team_players['element_type']==4, 'expected_goals_fraction']*expected_goals
        team_players.loc[team_players['element_type']==3, 'week'+str(current_week+week+1)+'_score_exp'] += 5* team_players.loc[team_players['element_type']==3, 'expected_goals_fraction']*expected_goals
        team_players.loc[team_players['element_type']==2, 'week'+str(current_week+week+1)+'_score_exp'] += 6* team_players.loc[team_players['element_type']==2, 'expected_goals_fraction']*expected_goals
        #team_players.loc[team_players['element_type']==1, 'week'+str(current_week+week+1)+'_score_exp'] += 6* team_players.loc[team_players['element_type']==1, 'expected_goals_fraction']*expected_goals

        #add to score for assists
        team_players.loc[:,'week'+str(current_week+week+1)+'_score_exp'] += 3 * team_players.loc[:,'expected_assists_fraction'] * expected_goals

        team_players.loc[team_players['element_type'] == 3, 'week' + str(current_week + week + 1) + '_score_exp'] +=(
                clean_sheet_prob * (team_players.loc[team_players['element_type'] == 3,'mins3']>60))
        team_players.loc[team_players['element_type'] == 2, 'week' + str(current_week + week + 1) + '_score_exp'] += (
                clean_sheet_prob * (team_players.loc[team_players['element_type'] == 2,'mins3']>60))
        team_players.loc[team_players['element_type'] == 1, 'week' + str(current_week + week + 1) + '_score_exp'] +=(
                clean_sheet_prob * (team_players.loc[team_players['element_type'] == 1,'mins3']>60))
    if id == 1:
        players2 = team_players
    else:
        players2 = pd.concat([players2,team_players], axis = 0)

players2['total_score_pred'] = 0
for week in range(current_week+1, current_week + 1 + weeks_ahead):
    players2['total_score_pred'] += players2['week' + str(week) + '_score_exp']
players2['value'] = players2['total_score_pred'] / players2['now_cost']

players2.to_csv('data/players_predicted_scores.csv', index = False)


# defenders
players2[players2['element_type']==2].sort_values('total_score_pred', ascending = False)[['second_name','team_name','now_cost', 'value','total_score_pred']].head(10)
players2[players2['element_type']==2].sort_values('value', ascending = False)[['second_name','team_name','now_cost', 'value','total_score_pred']].head(10)

players2[players2['element_type']==3].sort_values('total_score_pred', ascending = False)[['second_name','team_name','now_cost', 'value','total_score_pred']].head(10)
players2[players2['element_type']==3].sort_values('value', ascending = False)[['second_name','team_name','now_cost', 'value','total_score_pred']].head(10)

players2[players2['element_type']==4].sort_values('total_score_pred', ascending = False)[['second_name','team_name','now_cost', 'value','total_score_pred']].head(10)
players2[players2['element_type']==4].sort_values('value', ascending = False)[['second_name','team_name','now_cost', 'value','total_score_pred']].head(10)




players2.columns[:20]
players2['type']
players2['code']
players2['element_type'].head()

#Push?