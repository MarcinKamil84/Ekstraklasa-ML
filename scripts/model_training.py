# IMPORTING DATA

X_G_bal = pd.read_pickle('./data/X_G_bal.pkl')
X_Gl_bal = pd.read_pickle('./data/X_Gl_bal.pkl')


# FEATURES TYPES

cat_feats = ['home_or_away', 'season_ranks']

num_feats = ['team', 'home_or_away', 'opponent', 'czas_delta', 'season_points'] + [col for col in X_G_bal.columns if '_g_' in col] + [col for col in X_G_bal.columns if '_gl_' in col]


# MODEL TRAINING

# Points (win/draw/lose)

s_points = setup(data = X_G_bal, target = 'points',
                 ignore_features=['goals', 'goals_lost'],
                 silent=True,
                 categorical_features = cat_feats,
                 remove_multicollinearity = True)


best_points = compare_models(n_select = 3)

best_points_blended = blend_models(estimator_list = best_points, choose_better=True)

best_points_stacked = stack_models(estimator_list = best_points_blended, choose_better=True)

predict_model(best_points_stacked)

final_best_points_model = finalize_model(best_points_stacked)

save_model(final_best_points_model,'./models/final_best_points_model')

points_predictions = predict_model(final_best_points_model, data=ekstraklasa_test)


# Applying points predictions to train goals

X_G_bal['points'] = points_predictions['Label']
X_Gl_bal['points'] = points_predictions['Label']


# Home goals training

s_home_goals = setup(data = X_G_bal, target = 'goals',
                     ignore_features=['goals_lost'],
                     silent=True,
                     categorical_features = cat_feats,
                     remove_multicollinearity = True)


best_home_goals = compare_models(n_select = 3)

best_home_goals_blended = blend_models(estimator_list = best_home_goals, choose_better=True)

best_home_goals_stacked = stack_models(estimator_list = best_home_goals_blended, choose_better=True)

predict_model(best_home_goals_stacked)

final_best_home_goals_model = finalize_model(best_home_goals_stacked)

save_model(final_best_home_goals_model,'./models/final_best_home_goals_model')


# Away goals training

s_away_goals = setup(data = X_Gl_bal, target = 'goals_lost',
                     ignore_features=['goals'],
                     silent=True,
                     categorical_features = cat_feats,
                     remove_multicollinearity = True)


best_away_goals = compare_models(n_select = 3)

best_away_goals_blended = blend_models(estimator_list = best_away_goals, choose_better=True)

best_away_goals_stacked = stack_models(estimator_list = best_away_goals_blended, choose_better=True)

predict_model(best_away_goals_stacked)

final_best_away_goals_model = finalize_model(best_away_goals_stacked)

save_model(final_best_away_goals_model,'./models/final_best_away_goals_model')