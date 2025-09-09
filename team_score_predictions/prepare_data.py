import pandas as pd

def process_team_data_year(df):
    df_all = None
    for i in range(1, 21):
        df_home = df[df['team_h'] == i]
        df_home['home'] = 1
        df_home.rename(columns={'team_h_score': 'goals_scored', 'team_a_score': 'goals_conceded'}, inplace=True)
        # Want a rolling average of goals scored and conceded at home in the last 3 games
        df_home['rolling_avg_goals_scored_home'] = df_home['goals_scored'].rolling(3).mean().shift(1)
        df_home['rolling_avg_goals_conceded_home'] = df_home['goals_conceded'].rolling(3).mean().shift(1)


        df_away = df[df['team_a'] == i]
        df_away['away'] = 0
        df_away.rename(columns={'team_a_score': 'goals_scored', 'team_h_score': 'goals_conceded'}, inplace=True)
        # Want a rolling average of goals scored and conceded away in the last 3 games
        df_away['rolling_avg_goals_scored_away'] = df_away['goals_scored'].rolling(3).mean().shift(1)
        df_away['rolling_avg_goals_conceded_away'] = df_away['goals_conceded'].rolling(3).mean().shift(1)

        #Combine the two data sets
        df_both = pd.concat([df_home, df_away])
        #order by 'event'
        df_both.sort_values(by=['event'], inplace=True)
        df_both['rolling_avg_goals_conceded_away'] = df_both['rolling_avg_goals_conceded_away'].ffill()
        df_both['rolling_avg_goals_scored_away'] = df_both['rolling_avg_goals_scored_away'].ffill()
        df_both['rolling_avg_goals_scored_home'] = df_both['rolling_avg_goals_scored_home'].ffill()
        df_both['rolling_avg_goals_conceded_home'] = df_both['rolling_avg_goals_conceded_home'].ffill()
        df_both['team'] = i
        df_both = df_both[['team', 'event', 'rolling_avg_goals_conceded_away', 'rolling_avg_goals_scored_away', 'rolling_avg_goals_scored_home',
        'rolling_avg_goals_conceded_home']]
        df_all = pd.concat([df_all, df_both])

    # Now look at the fixtures, combine the available team data for each game to get Xs and ys
    n = df.shape[0]
    df_final = None
    for i in range(n):
        row_dat = df[['event','team_h', 'team_a', 'team_h_score', 'team_a_score']].iloc[i:(i+1)]

        row_dat = pd.merge(row_dat, df_all, how='left', left_on=['event', 'team_h'], right_on=['event', 'team'])
        row_dat.drop(['team'], axis=1, inplace=True)

        row_dat = pd.merge(row_dat, df_all, how='left', left_on=['event', 'team_a'], right_on=['event', 'team'])
        row_dat.drop(['team', 'team_a', 'team_h', 'event'], axis=1, inplace=True)
        df_final = pd.concat([df_final, row_dat])

    df_final.dropna(inplace=True)
    return df_final
