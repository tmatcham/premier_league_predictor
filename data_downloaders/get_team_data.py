import pandas as pd

# URL of the CSV file (example)
url_head = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/20"
url_tail = "/fixtures.csv"

def get_team_data(year):
    year_string = str(year) + '-' +str(year+1)
    url = url_head + year_string + url_tail
    df = pd.read_csv(url)
    return df

