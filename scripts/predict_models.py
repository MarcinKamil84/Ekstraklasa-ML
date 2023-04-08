from pycaret.classification import *
import pickle

# Loading models and variables

final_best_points_model = load_model('./models/points_final')
final_best_home_model = load_model('./models/goals_final')

with open("./data/variables/teams_dict.pkl", "rb") as file:
    teams_dict = pickle.load(file)

ekstraklasa_test = pd.read_pickle('./data/inputs/ekstraklasa_test.pkl')

points_predictions = predict_model(final_best_points_model, data=ekstraklasa_test)
home_predictions = predict_model(final_best_home_model, data=ekstraklasa_test)


# COMBINING THE RESULTS

results_df = pd.DataFrame()

results_df['dom'] = home_predictions['team']
results_df['wyjazd'] = home_predictions['opponent']

results_df['punkty'] = points_predictions['Label']
results_df['punkty_score'] = round(points_predictions['Score'], 2)

results_df['bramki_dom'] = home_predictions['Label']
results_df['bramki_dom_score'] = round(home_predictions['Score'], 2)

results_df['home_or_away'] = home_predictions['home_or_away']

results_df_combo = results_df.merge(results_df, how='left',
                                    left_on=['dom', 'wyjazd'],
                                    right_on=['wyjazd', 'dom'],
                                    )


results_df_final = pd.DataFrame()

results_df_final['home_or_away'] = results_df_combo['home_or_away_x']

results_df_final['dom'] = results_df_combo['dom_x']
results_df_final['wyjazd'] = results_df_combo['wyjazd_x']

results_df_final['wynik'] = results_df_combo['punkty_x']
results_df_final['wynik_score'] = results_df_combo['punkty_score_x']

results_df_final['bramki_dom'] = results_df_combo['bramki_dom_x']
results_df_final['bramki_dom_score'] = results_df_combo['bramki_dom_score_x']

results_df_final['bramki_wyjazd'] = results_df_combo['bramki_dom_y']
results_df_final['bramki_wyjazd_score'] = results_df_combo['bramki_dom_score_y']

results_df_final = results_df_final[results_df_final['home_or_away']==True]

results_df_final.drop(labels = ['home_or_away'], axis = 1, inplace = True)
results_df_final.reset_index(drop=True, inplace = True)

results_df_final['prob_sum'] = round((results_df_final.wynik_score + results_df_final.bramki_dom_score + results_df_final.bramki_wyjazd_score), 2)

results_df_final['prob_avg'] = round(((results_df_final.wynik_score + results_df_final.bramki_dom_score + results_df_final.bramki_wyjazd_score) / 3), 2)

results_df_final.sort_values('wynik_score', ascending=False, inplace = True)

results_df_final.reset_index(drop=True, inplace = True)

inv_teams = {v: k for k, v in teams_dict.items()}

results_df_final['dom'] = results_df_final['dom'].replace(inv_teams)
results_df_final['wyjazd'] = results_df_final['wyjazd'].replace(inv_teams)


# SAVING RESULTS

results_df_final.to_pickle('./data/outputs/preds/results_final.pkl')
