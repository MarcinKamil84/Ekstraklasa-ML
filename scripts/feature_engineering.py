# IMPORTS

import pandas as pd
import numpy as np
import re
from datetime import date, datetime, timedelta
from scipy.stats import mode


# LOADING DATA

filepath = './data/ekstraklasa.csv'
cols = ['data', 'dom', 'wyjazd', 'wynik']

ekstraklasa = pd.read_csv(filepath, usecols=cols)

ekstraklasa.columns = ['gameday', 'home', 'away', 'wynik']

ekstraklasa['gameday'] = ekstraklasa['gameday'].fillna(method='ffill')


# INITIAL CLEANING

ekstraklasa['gameday'] = pd.to_datetime(ekstraklasa['gameday'], format='%d/%m/%Y')

ekstraklasa['wynik'] = ekstraklasa['wynik'].str.replace(r'\s\(\d+\:\d+\)','', regex=True)

ekstraklasa['wynik'] = ekstraklasa['wynik'].str.replace('-:-','99:99', regex=True)

ekstraklasa[['wynik_home', 'wynik_away']] = ekstraklasa['wynik'].str.split(':', expand=True)

patternDel = '^(\D)|.{3,}$'

filter = ekstraklasa['wynik_home'].str.contains(patternDel)
ekstraklasa = ekstraklasa[~filter]

filter = ekstraklasa['wynik_away'].str.contains(patternDel)
ekstraklasa = ekstraklasa[~filter]

ekstraklasa = ekstraklasa.sort_values(by=['gameday'], ascending=False)


# Defining win/draw/lose

ekstraklasa = ekstraklasa.astype({'wynik_home': 'int', 'wynik_away': 'int'})

punkty_dom = []
punkty_wyjazd = []

for row in ekstraklasa.itertuples():
    
    wynik_home = row.wynik_home
    wynik_away = row.wynik_away
    
    if wynik_home > wynik_away:
        stan_dom = 2
        stan_wyjazd = 0
    elif wynik_home < wynik_away:
        stan_dom = 0
        stan_wyjazd = 2
    else:
        stan_dom = 1
        stan_wyjazd = 1
        
    punkty_dom.append(stan_dom)
    punkty_wyjazd.append(stan_wyjazd)
    
ekstraklasa['points_home'] = punkty_dom
ekstraklasa['points_away'] = punkty_wyjazd

ekstraklasa = ekstraklasa.drop('wynik', axis=1)


# Creating new dataframes for ML

ekstraklasa_ml = ekstraklasa.copy()

czas_first = ekstraklasa_ml['gameday'].max()
czas_delta = []


# Convert years to delta value between first and current date

for rok in ekstraklasa.itertuples():
    
    deltas = (czas_first.year - rok.gameday.year)
        
    czas_delta.append(deltas)
    
ekstraklasa_ml['czas_delta'] = czas_delta

ekstraklasa_ml.sort_values(by='gameday', inplace=True)


# Creating one table with all teams

main_columns_1 = ['gameday', 'czas_delta'] + [col for col in ekstraklasa_ml.columns if 'home' in col]
main_columns_2 = ['gameday', 'czas_delta'] + [col for col in ekstraklasa_ml.columns if 'away' in col]

main_df_1 = ekstraklasa_ml[main_columns_1]
main_df_2 = ekstraklasa_ml[main_columns_2]

main_df_1['home_or_away'] = True
main_df_2['home_or_away'] = False

main_df_1.columns = ['gameday', 'czas_delta', 'team', 'goals', 'points', 'home_or_away']
main_df_2.columns = ['gameday', 'czas_delta', 'team', 'goals', 'points', 'home_or_away']

main_df_1['opponent'] = main_df_2['team']
main_df_2['opponent'] = main_df_1['team']

main_df_1['goals_lost'] = main_df_2['goals']
main_df_2['goals_lost'] = main_df_1['goals']

main_df = pd.concat([main_df_1, main_df_2], ignore_index=True).fillna(0).rename_axis('MyIdx').sort_values(by=['gameday','MyIdx'], ascending=True).reset_index(drop=True)


# Goals for average

main_df['last_10_g_mean'] = round(main_df.groupby('team')['goals']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).mean()), 4)

main_df['last_10_g_mean_ha'] = round(main_df.groupby(['team','home_or_away'])['goals']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).mean()), 4)

main_df['last_10_g_mean_pair'] = round(main_df.groupby(['team','opponent'])['goals']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).mean()), 4)

main_df['last_3_g_mean'] = round(main_df.groupby('team')['goals']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).mean()), 4)

main_df['last_3_g_mean_ha'] = round(main_df.groupby(['team','home_or_away'])['goals']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).mean()), 4)

main_df['last_3_g_mean_pair'] = round(main_df.groupby(['team','opponent'])['goals']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).mean()), 4)


# Goals against average

main_df['last_10_gl_mean'] = round(main_df.groupby('team')['goals_lost']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).mean()), 4).fillna(0)

main_df['last_10_gl_mean_ha'] = round(main_df.groupby(['team','home_or_away'])['goals_lost']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).mean()), 4).fillna(0)

main_df['last_10_gl_mean_pair'] = round(main_df.groupby(['team','opponent'])['goals_lost']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).mean()), 4).fillna(0)

main_df['last_3_gl_mean'] = round(main_df.groupby('team')['goals_lost']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).mean()), 4).fillna(0)

main_df['last_3_gl_mean_ha'] = round(main_df.groupby(['team','home_or_away'])['goals_lost']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).mean()), 4).fillna(0)

main_df['last_3_gl_mean_pair'] = round(main_df.groupby(['team','opponent'])['goals_lost']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).mean()), 4).fillna(0)


# Points sum

main_df['last_10_p_sum'] = round(main_df.groupby('team')['points']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).sum()), 4).fillna(0)

main_df['last_10_p_sum_ha'] = round(main_df.groupby(['team','home_or_away'])['points']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).sum()), 4).fillna(0)

main_df['last_10_p_sum_pair'] = round(main_df.groupby(['team','opponent'])['points']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).sum()), 4).fillna(0)

main_df['last_3_p_sum'] = round(main_df.groupby('team')['points']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).sum()), 4).fillna(0)

main_df['last_3_p_sum_ha'] = round(main_df.groupby(['team','home_or_away'])['points']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).sum()), 4).fillna(0)

main_df['last_3_p_sum_pair'] = round(main_df.groupby(['team','opponent'])['points']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).sum()), 4).fillna(0)

main_df['sum_all'] = main_df.groupby(['team'])['points'].cumsum()


# Goals mode

main_df['last_10_g_mode'] = main_df.groupby('team')['goals']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')

main_df['last_10_g_mode_ha'] = main_df.groupby(['team','home_or_away'])['goals']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')

main_df['last_10_g_mode_pair'] = main_df.groupby(['team','opponent'])['goals']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')

main_df['last_3_g_mode'] = main_df.groupby('team')['goals']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')

main_df['last_3_g_mode_ha'] = main_df.groupby(['team','home_or_away'])['goals']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')

main_df['last_3_g_mode_pair'] = main_df.groupby(['team','opponent'])['goals']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')

main_df['last_10_gl_mode'] = main_df.groupby('team')['goals_lost']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')

main_df['last_10_gl_mode_ha'] = main_df.groupby(['team','home_or_away'])['goals_lost']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')

main_df['last_10_gl_mode_pair'] = main_df.groupby(['team','opponent'])['goals_lost']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')

main_df['last_3_gl_mode'] = main_df.groupby('team')['goals_lost']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')

main_df['last_3_gl_mode_ha'] = main_df.groupby(['team','home_or_away'])['goals_lost']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')

main_df['last_3_gl_mode_pair'] = main_df.groupby(['team','opponent'])['goals_lost']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')


# Season ranking

main_df['season_points'] = round(main_df.groupby('team')['points']    .transform(lambda x: x.shift(periods=1, axis=0).rolling(24, min_periods=1).sum()), 4).fillna(0)
main_df['season_points'].astype('int32')

main_df['season_ranks'] = pd.cut(main_df['season_points'], bins=18, labels=np.arange(1, 19))
main_df['season_ranks'].replace([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18], [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18], inplace=True)
main_df['season_ranks'] = main_df['season_ranks'].astype('int32')


# Filling NaN

main_df.fillna(0, inplace=True)

# Turning team names into dictionary

teams = main_df.loc[(main_df['goals'] < 10)].sort_values('sum_all', ascending=True)['team'].unique()

teams_dict = {i: j for j, i in enumerate(teams)}

main_df['team'] = main_df['team'].replace(teams_dict)
main_df['opponent'] = main_df['opponent'].replace(teams_dict)


# DATA PREPARATION FOR ML

# Create test df

ekstraklasa_test = main_df.loc[(main_df['gameday'] >= start_d) & (main_df['gameday'] <= end_d)]


# Create train df

ekstraklasa_train = main_df.loc[(main_df['gameday'] < start_d) & (main_df['goals'] < 6) & (main_df['goals_lost'] < 6)]
ekstraklasa_train = ekstraklasa_train.loc[(main_df['czas_delta'] != main_df['czas_delta'].max())]


# FIXING CLASS IMBALANCE

ekstraklasa_train.drop(['gameday'], inplace=True, axis=1)

X_G = ekstraklasa_train.drop(['goals'], axis=1)
y_g = ekstraklasa_train.loc[:,'goals']

ros = RandomOverSampler(random_state=42)

X_G_bal, y_g_bal = ros.fit_resample(X_G, y_g)

X_G_bal['goals'] = y_g_bal

X_Gl = ekstraklasa_train.drop(['goals_lost'], axis=1)
y_gl = ekstraklasa_train.loc[:,'goals_lost']

ros = RandomOverSampler(random_state=42)

X_Gl_bal, y_gl_bal = ros.fit_resample(X_Gl, y_gl)

X_Gl_bal['goals_lost'] = y_gl_bal


# EXPORTING DATAFRAMES FOR ML

X_G_bal.to_pickle('./data/X_G_bal.pkl')
X_Gl_bal.to_pickle('./data/X_Gl_bal.pkl')