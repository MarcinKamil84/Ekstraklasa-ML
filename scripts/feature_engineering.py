import pandas as pd
import numpy as np
import re
from scipy.stats import mode
import pickle
from imblearn.over_sampling import RandomOverSampler


# LOADING DATA
print('Loading data...')

filepath = './data/inputs/ekstraklasa_historical_results.csv'
cols = ['data', 'dom', 'wyjazd', 'wynik']

ekstraklasa = pd.read_csv(filepath, usecols=cols)

ekstraklasa.columns = ['gameday', 'home', 'away', 'wynik']

ekstraklasa['gameday'] = ekstraklasa['gameday'].fillna(method='ffill')


# INITIAL CLEANING
print('Initial cleaning...')

ekstraklasa['gameday'] = pd.to_datetime(ekstraklasa['gameday'], format='%d/%m/%Y')

ekstraklasa['wynik'] = ekstraklasa['wynik'].str.replace('-:-','99:99', regex=True)

wynik_temp = []
for value in ekstraklasa['wynik']:
    if re.search(r'^(\d+:\d+)', value) is None:
        wynik_temp.append(np.nan)
    else:
        wynik_temp.append(re.search(r'^(\d+:\d+)', value).group())

ekstraklasa['wynik'] = wynik_temp

ekstraklasa.dropna(axis=0, subset='wynik', inplace=True)

ekstraklasa[['wynik_home', 'wynik_away']] = ekstraklasa['wynik'].str.split(':', expand=True)

ekstraklasa = ekstraklasa.sort_values(by=['gameday'], ascending=False)


# Defining win/draw/lose
print('Defining win/draw/lose...')

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

main_df[main_df['home_or_away'] == False]['points'].replace([0,2], [2,0], inplace=True)


# Goals for average
print('Engineering goals...')

main_df['last_10_g_mean'] = round(main_df.groupby('team')['goals']
                                  .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).mean()), 4)

main_df['last_10_g_mean_ha'] = round(main_df.groupby(['team','home_or_away'])['goals']
                                     .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).mean()), 4)

main_df['last_10_g_mean_pair'] = round(main_df.groupby(['team','opponent'])['goals']
                                       .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).mean()), 4)

main_df['last_3_g_mean'] = round(main_df.groupby('team')['goals']
                                 .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).mean()), 4)

main_df['last_3_g_mean_ha'] = round(main_df.groupby(['team','home_or_away'])['goals']
                                    .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).mean()), 4)

main_df['last_3_g_mean_pair'] = round(main_df.groupby(['team','opponent'])['goals']
                                      .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).mean()), 4)


# Goals against average

main_df['last_10_gl_mean'] = round(main_df.groupby('team')['goals_lost']
                                   .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).mean()), 4).fillna(0)

main_df['last_10_gl_mean_ha'] = round(main_df.groupby(['team','home_or_away'])['goals_lost']
                                      .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).mean()), 4).fillna(0)

main_df['last_10_gl_mean_pair'] = round(main_df.groupby(['team','opponent'])['goals_lost']
                                        .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).mean()), 4).fillna(0)

main_df['last_3_gl_mean'] = round(main_df.groupby('team')['goals_lost']
                                  .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).mean()), 4).fillna(0)

main_df['last_3_gl_mean_ha'] = round(main_df.groupby(['team','home_or_away'])['goals_lost']
                                     .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).mean()), 4).fillna(0)

main_df['last_3_gl_mean_pair'] = round(main_df.groupby(['team','opponent'])['goals_lost']
                                       .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).mean()), 4).fillna(0)


# Goals mode

main_df['last_10_g_mode'] = main_df.groupby('team')['goals'].transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')

main_df['last_10_g_mode_ha'] = main_df.groupby(['team','home_or_away'])['goals'].transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')

main_df['last_10_g_mode_pair'] = main_df.groupby(['team','opponent'])['goals'].transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')

main_df['last_3_g_mode'] = main_df.groupby('team')['goals'].transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')

main_df['last_3_g_mode_ha'] = main_df.groupby(['team','home_or_away'])['goals'].transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')

main_df['last_3_g_mode_pair'] = main_df.groupby(['team','opponent'])['goals'].transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')

main_df['last_10_gl_mode'] = main_df.groupby('team')['goals_lost'].transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')

main_df['last_10_gl_mode_ha'] = main_df.groupby(['team','home_or_away'])['goals_lost'].transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')

main_df['last_10_gl_mode_pair'] = main_df.groupby(['team','opponent'])['goals_lost'].transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')

main_df['last_3_gl_mode'] = main_df.groupby('team')['goals_lost'].transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')

main_df['last_3_gl_mode_ha'] = main_df.groupby(['team','home_or_away'])['goals_lost'].transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')

main_df['last_3_gl_mode_pair'] = main_df.groupby(['team','opponent'])['goals_lost'].transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).apply(lambda x: mode(x)[0])).fillna(0).astype('int32')


# Points sum
print('Engineering points...')

main_df['last_10_p_sum'] = round(main_df.groupby('team')['points']
                                 .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).sum()), 4).fillna(0)

main_df['last_10_p_sum_ha'] = round(main_df.groupby(['team','home_or_away'])['points']
                                    .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).sum()), 4).fillna(0)

main_df['last_10_p_sum_pair'] = round(main_df.groupby(['team','opponent'])['points']
                                      .transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).sum()), 4).fillna(0)

main_df['last_3_p_sum'] = round(main_df.groupby('team')['points']
                                .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).sum()), 4).fillna(0)

main_df['last_3_p_sum_ha'] = round(main_df.groupby(['team','home_or_away'])['points']
                                   .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).sum()), 4).fillna(0)

main_df['last_3_p_sum_pair'] = round(main_df.groupby(['team','opponent'])['points']
                                     .transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).sum()), 4).fillna(0)

main_df['sum_all'] = main_df.groupby(['team'])['points'].cumsum()


# Number of wins, draws and defeats in the last 34 games
print('Engineering walking averages...')

main_df['season_wins'] = main_df.replace([0, 1], np.nan).groupby(['team'])['points'].transform(lambda x: x.shift(periods=1, axis=0).rolling(34, min_periods=1).count() / 34).fillna(0)
main_df['season_draws'] = main_df.replace([0, 2], np.nan).groupby(['team'])['points'].transform(lambda x: x.shift(periods=1, axis=0).rolling(34, min_periods=1).count() / 34).fillna(0)
main_df['season_defeats'] = main_df.replace([1, 2], np.nan).groupby(['team'])['points'].transform(lambda x: x.shift(periods=1, axis=0).rolling(34, min_periods=1).count() / 34).fillna(0)


# Number of wins, draws and defeats in the last 10 games

main_df['last10_wins'] = main_df.replace([0, 1], np.nan).groupby(['team'])['points'].transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).count() / 10).fillna(0)
main_df['last10_draws'] = main_df.replace([0, 2], np.nan).groupby(['team'])['points'].transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).count() / 10).fillna(0)
main_df['last10_defeats'] = main_df.replace([1, 2], np.nan).groupby(['team'])['points'].transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).count() / 10).fillna(0)


# Win, draw or defeat in the last game

main_df['last_wins'] = main_df.replace([0, 1], np.nan).groupby(['team'])['points'].transform(lambda x: x.shift(periods=1, axis=0).rolling(1, min_periods=1).count()).fillna(0)
main_df['last_draws'] = main_df.replace([0, 2], np.nan).groupby(['team'])['points'].transform(lambda x: x.shift(periods=1, axis=0).rolling(1, min_periods=1).count()).fillna(0)
main_df['last_defeats'] = main_df.replace([1, 2], np.nan).groupby(['team'])['points'].transform(lambda x: x.shift(periods=1, axis=0).rolling(1, min_periods=1).count()).fillna(0)


# Number of wins, draws and defeats in the last 10 home/away games

main_df['last10_ha_wins'] = main_df.replace([0, 1], np.nan).groupby(['team', 'home_or_away'])['points'].transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).count() / 10).fillna(0)
main_df['last10_ha_draws'] = main_df.replace([0, 2], np.nan).groupby(['team', 'home_or_away'])['points'].transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).count() / 10).fillna(0)
main_df['last10_ha_defeats'] = main_df.replace([1, 2], np.nan).groupby(['team', 'home_or_away'])['points'].transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).count() / 10).fillna(0)


# Number of wins, draws and defeats against the same opponent in the last 10 games

main_df['same_team_wins'] = main_df.replace([0, 1], np.nan).groupby(['team', 'opponent'])['points'].transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).count() / 10).fillna(0)
main_df['same_team_draws'] = main_df.replace([0, 2], np.nan).groupby(['team', 'opponent'])['points'].transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).count() / 10).fillna(0)
main_df['same_team_defeats'] = main_df.replace([1, 2], np.nan).groupby(['team', 'opponent'])['points'].transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).count() / 10).fillna(0)


# Season ranking team

main_df['season_points'] = round(main_df.groupby('team')['points'].transform(lambda x: x.shift(periods=1, axis=0).rolling(34, min_periods=1).sum()), 4).fillna(0).astype('int32')
main_df['season_points_opponent'] = round(main_df.groupby('opponent')['points'].transform(lambda x: x.shift(periods=1, axis=0).rolling(34, min_periods=1).sum()), 4).fillna(0).astype('int32')

main_df['season_points_difference'] = main_df['season_points'] - main_df['season_points_opponent']

main_df['season_points_difference_rank'] = pd.cut(main_df['season_points_difference'], bins=20, include_lowest=True,
                                                  labels=np.arange(-10, 10, 1))


# Number of wins, draws and defeats against similar opponent in the last 10 games

main_df['sim_team_wins'] = main_df.replace([0, 1], np.nan).groupby(['opponent', 'season_points_difference_rank'])['points'].transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).count() / 10).fillna(0)
main_df['sim_team_draws'] = main_df.replace([0, 2], np.nan).groupby(['opponent', 'season_points_difference_rank'])['points'].transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).count() / 10).fillna(0)
main_df['sim_team_defeats'] = main_df.replace([1, 2], np.nan).groupby(['opponent', 'season_points_difference_rank'])['points'].transform(lambda x: x.shift(periods=1, axis=0).rolling(10, min_periods=1).count() / 10).fillna(0)


# NLP ENGINEERING

print('Engineering nlp data...')

teams = main_df.loc[(main_df['goals'] < 10)].sort_values('sum_all', ascending=True)['team'].unique()
teams_small = list(main_df.loc[(main_df['goals'] < 10)].sort_values('sum_all', ascending=True)['team'].str.lower().unique())

news_path = './data/news/articles_gol24_nlp.json'
news_df = pd.read_json(news_path)

pat = '|'.join(r"\b{}\b".format(x) for x in teams_small)
news_df = news_df[news_df['tags'].str.contains(pat)]
news_df['date'] = pd.to_datetime(news_df['date'], format='%d/%m/%Y')
news_df = news_df.sort_values('date')

main_df['team_lower'] = main_df['team'].str.lower()

news_by_team_df = pd.DataFrame(columns=news_df.columns)

# Creating table with articles for separate teams

for i, row in news_df.iterrows():

    for t in teams_small:

        if t in row['tags']:

            news_by_team_df.loc[len(news_by_team_df)] = row
            news_by_team_df.loc[len(news_by_team_df)-1, 'team_tag'] = t


print(news_by_team_df.team_tag.nunique())

print('Compiling nlp features...')

main_df['news_pos'] = news_by_team_df.groupby('team_tag')['rate_pos'].transform(lambda x: x.shift(periods=1, axis=0).rolling(1, min_periods=1).mean()).astype('float32')
main_df['news_neg'] = news_by_team_df.groupby('team_tag')['rate_neg'].transform(lambda x: x.shift(periods=1, axis=0).rolling(1, min_periods=1).mean()).astype('float32')

main_df['news_pos_3'] = news_by_team_df.groupby('team_tag')['rate_pos'].transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).mean()).astype('float32')
main_df['news_neg_3'] = news_by_team_df.groupby('team_tag')['rate_neg'].transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).mean()).astype('float32')

main_df['news_pos_7'] = news_by_team_df.groupby('team_tag')['rate_pos'].transform(lambda x: x.shift(periods=1, axis=0).rolling(7, min_periods=1).mean()).astype('float32')
main_df['news_neg_7'] = news_by_team_df.groupby('team_tag')['rate_neg'].transform(lambda x: x.shift(periods=1, axis=0).rolling(7, min_periods=1).mean()).astype('float32')

main_df['news_pos_max'] = news_by_team_df.groupby('team_tag')['rate_pos'].transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).max()).astype('float32')
main_df['news_neg_max'] = news_by_team_df.groupby('team_tag')['rate_neg'].transform(lambda x: x.shift(periods=1, axis=0).rolling(3, min_periods=1).max()).astype('float32')

columns_to_correct = ['news_pos', 'news_neg', 'news_pos_3', 'news_neg_3', 'news_pos_7', 'news_neg_7']

main_df[columns_to_correct] = main_df[columns_to_correct].fillna(value=False)

print(main_df['news_pos'].nunique())
print(main_df['news_pos'].sample(10))

# Filling NaN

main_df.fillna(0, inplace=True)

main_df.drop(['team_lower'], axis=1)

# Turning team names into dictionary
print('Turning team names into dictionary...')

teams = main_df.loc[(main_df['goals'] < 10)].sort_values('sum_all', ascending=True)['team'].unique()

teams_dict = {i: j for j, i in enumerate(teams)}

main_df['team'] = main_df['team'].replace(teams_dict)
main_df['opponent'] = main_df['opponent'].replace(teams_dict)

with open('./data/variables/teams_dict.pkl', 'wb') as file:
    pickle.dump(teams_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


# DATA PREPARATION FOR ML

start_d_file = open("data/variables/start_d.txt", 'r')
start_d_text = start_d_file.read()
start_d_file.close()

end_d_file = open("data/variables/end_d.txt", 'r')
end_d_text = end_d_file.read()
end_d_file.close()

start_d = np.datetime64(str(start_d_text))
end_d = np.datetime64(str(end_d_text))


# Create test df

ekstraklasa_test = main_df.loc[(main_df['gameday'] >= start_d) & (main_df['gameday'] <= end_d)]


# Create train df

ekstraklasa_train = main_df.loc[(main_df['gameday'] != 0) & (main_df['gameday'] < start_d) & (main_df['goals'] < 6) & (main_df['goals_lost'] < 6)]
ekstraklasa_train = ekstraklasa_train.loc[(main_df['czas_delta'] != main_df['czas_delta'].max())]
ekstraklasa_train = ekstraklasa_train.loc[(ekstraklasa_train['news_pos'] != False)]

# FIXING CLASS IMBALANCE
print('Fixing class imbalance...')

ekstraklasa_train.drop(['gameday'], inplace=True, axis=1)

X_G = ekstraklasa_train.drop(['goals'], axis=1)
y_g = ekstraklasa_train.loc[:,'goals']

ros = RandomOverSampler(random_state=42)

ekstraklasa_balanced_train, y_g_bal = ros.fit_resample(X_G, y_g)

ekstraklasa_balanced_train['goals'] = y_g_bal


# EXPORTING DATAFRAMES FOR ML
print('Exporting data...')

ekstraklasa_train.to_pickle('./data/inputs/ekstraklasa_train.pkl')
ekstraklasa_test.to_pickle('./data/inputs/ekstraklasa_test.pkl')
