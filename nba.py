import json
import pandas as pd
import numpy as np

# Received JSON filepath to 1st Half & 3rd Quarter Stats
# Returns 2 DataFrames - First Half Stats, 3rd Quarter Stats

# Extra columns returned by stats.nba.com that won't be needed
extra_initial_columns = ['SEASON_YEAR', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_DATE']

# Overlapping columns from 1st Half & 3rd Quarter stats that are unique to a team&game
columns_to_merge_games = ['TEAM_ID', 'GAME_ID', 'MATCHUP', 'WL']

#####################################
##                                 ##
## LOADING & PARSING JSON          ##   
##                                 ##
#####################################

def load_json(json_path_half, json_path_3q):
    json_half = json.load(open(json_path_half))
    json_3q = json.load(open(json_path_3q))
    columns = columns_from_json(json_half)
    rows_half = rows_from_json(json_half)
    rows_3q = rows_from_json(json_3q)
    df_half = pd.DataFrame(rows_half, columns=columns)
    df_3q = pd.DataFrame(rows_3q, columns=columns)
    return df_half, df_3q

# stats.nba.com returns keys/headers in 'headers'
def columns_from_json(json):
    return json['resultSets'][0]['headers']

# stats.nba.com returns game data in 'rowSet'
def rows_from_json(json):
    return json['resultSets'][0]['rowSet']

#####################################
##                                 ##
## REMOVING COLUMNS & ROWS         ##   
##                                 ##
#####################################

# remove season-aggregated stats like "RANK"
def remove_extra_columns(df):
    df = df[df.columns.drop(list(df.filter(regex='_RANK')))]
    df = df.drop(extra_initial_columns, axis=1)
    return df

# remove columns that are not statistical in nature and extra
def remove_extra_initial_columns(df):
    df = df.drop(extra_initial_columns, axis=1)
    return df

def drop_3q_half_columns(df):
    df = df[df.columns.drop(list(df.filter(regex='3Q_')))]
    df = df[df.columns.drop(list(df.filter(regex='HALF_')))]
    return df

def drop_columns_used_in_feature_engineering(df):
    extra_columns = ['FGM', 'FGA', 'FG_PCT', 
                     'FG3M', 'FG3A', 'FG3_PCT', 
                     'FTM', 'FTA', 'FT_PCT',
                     'OPP_FGM', 'OPP_FGA',
                     'OPP_FG3M', 'OPP_FG3A',
                     'OPP_FTM', 'OPP_FTA',
                     'PF', 'OPP_PF',
                     'OREB', 'DREB', 'OPP_OREB', 'OPP_DREB',
                     'TOV', 'OPP_TOV', 'STL', 'OPP_STL',
                     'PTS', 'OPP_PTS', 'REB', 'OPP_REB',
                     'OPP_PLUS_MINUS', 'BLKA', 'OPP_BLKA', 'PFD', 'OPP_PFD', 
                     'GAME_ID', 'TEAM_ID', 'MIN', 'OPP_MIN', 'HOME'
                    ]
    df = df.drop(extra_columns, axis=1)
    return df

def drop_unused_rows(df):
    df = df.dropna()
    return df

#####################################
##                                 ##
## RENAME, SORT & MERGE            ##   
##                                 ##
#####################################

def set_home_column(df):
    df['HOME'] = pd.np.where(df.MATCHUP.str.contains("vs."), 1, 0)
    return df

def sort_df_by_game(df):
    df = df.sort_values(['GAME_ID'])
    return df

# Prefix columns for 3Q data with "3Q_", Half data with "HALF_
# Remove prefix from columns that will be used to merge
def rename_stats_with_period(df, prefix):
    df = df.add_prefix(prefix)
    if prefix == "3Q_":
        df = df.rename(index=str, columns={"3Q_TEAM_ID": "TEAM_ID",
                                           "3Q_GAME_ID": "GAME_ID",
                                           "3Q_MATCHUP": "MATCHUP",
                                           "3Q_WL": "WL",
                                          })
    else: # prefix assumed to be "HALF_"
        df = df.rename(index=str, columns={"HALF_TEAM_ID": "TEAM_ID",
                                           "HALF_GAME_ID": "GAME_ID",
                                           "HALF_MATCHUP": "MATCHUP",
                                           "HALF_WL": "WL",
                                          })
    return df
    
# merges 3Q & Half Dataframes on columns that uniquely identify team/game
def merge_dataframes(df1, df2):
    df = df1.merge(df2, how='inner', on=columns_to_merge_games)
    return df

#####################################
##                                 ##
## STAT CALCULATIONS .             ##   
##                                 ##
#####################################

# Add Half & 3Q Stats to get totals at end of 3rd Quarter
def combine_3q_half_stats(df):
    columns_to_combine = ['MIN', 'FGM', 'FGA', 'FG3M',  'FG3A', 'FTM',  'FTA',  
                  'OREB',  'DREB',  'REB',  'AST',  'TOV',  'STL',  
                  'BLK',  'BLKA',  'PF',  'PFD',  'PTS', 'PLUS_MINUS']
    for col in columns_to_combine:
        df[col] = df["HALF_" + col] + df["3Q_" + col]

    return df

# FG%, FG3%, & FT% recalculations since attempts/made have changed with combining of stats
def recalculate_combined_stats(df):
    df['FG_PCT'] = df['FGM'] / df['FGA']
    df['FG3_PCT'] = df['FG3M'] / df['FG3A']
    df['FT_PCT'] = df['FTM'] / df['FTA']
    return df


def add_opponent_stats_to_df(df):
    opp_columns = ['MIN', 'FGM', 'FGA', 'FG3M',  'FG3A', 'FTM',  'FTA',  
                  'OREB',  'DREB',  'REB',  'AST',  'TOV',  'STL',  
                  'BLK',  'BLKA',  'PF',  'PFD',  'PTS', 'PLUS_MINUS']

    # iterate through dataframe, every other index (since, rows ordered by game id)
    for i in range(0, len(df), 2):
        # get game id of row with current index
        gid = df.iloc[i]['GAME_ID']
    
        for col in opp_columns:
            # locate row of away team (on matching game id & home == 0) and get column value
            away_stat = df.loc[(df['GAME_ID'] == gid) & (df['HOME'] == 0), col].values[0]
            new_col = "OPP_" + col
            # assign opponent stat is new column in current row (home team)
            df.loc[(df['GAME_ID'] == gid) & (df['HOME'] == 1), new_col] = away_stat
    
    return df

def calculate_four_factor_statistics(df):
    # Calculate Four-Factor stats for home team
    df["EFG_PCT"] = (df.FGM + 0.5*df.FG3M)/df.FGA
    df["TOV_PCT"] = df.TOV/(df.FGA - df.OREB + df.TOV + 0.4*df.FTA)
    df["OREB_PCT"] = df.OREB / (df.OREB + df.OPP_DREB)
    df["FT_RATE"] = df.FTA / df.FGA

    # Calculate Four-Factor stats for away team (OPP)
    df["OPP_EFG_PCT"] = (df.OPP_FGM + 0.5*df.OPP_FG3M)/df.OPP_FGA
    df["OPP_TOV_PCT"] = df.OPP_TOV/(df.OPP_FGA - df.OPP_OREB + df.OPP_TOV + 0.4*df.OPP_FTA)
    df["OPP_FT_RATE"] = df.OPP_FTA / df.OPP_FGA
    df["DREB_PCT"] = df.DREB / (df.OPP_FGA - df.OPP_FGM)
    
    return df

