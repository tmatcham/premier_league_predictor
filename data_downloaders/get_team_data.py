import pandas as pd

# URL of the CSV file (example)
url_head = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/20"

def get_team_data(year):
    year_string = str(year) + '-' +str(year+1)
    url = url_head + year_string + "/fixtures.csv"
    df = pd.read_csv(url)
    return df


def get_teams(year):
    year_string = str(year) + '-' +str(year+1)
    url = url_head + year_string + "/teams.csv"
    df = pd.read_csv(url)
    return df
