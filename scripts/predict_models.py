# Loading the models

final_best_points_model = load_model('./models/final_best_points_model')
final_best_home_model = load_model('./models/final_best_home_goals_model')
final_best_away_model = load_model('./models/final_best_away_goals_model')

points_predictions = predict_model(final_best_points_model, data=ekstraklasa_test)
home_predictions = predict_model(final_best_home_model, data=ekstraklasa_test)
away_predictions = predict_model(final_best_away_model, data=ekstraklasa_test)


# COMBINING THE RESULTS

results_df = pd.DataFrame()

results_df['dom'] = home_predictions['team']
results_df['wyjazd'] = away_predictions['opponent']

results_df['punkty'] = points_predictions['Label']
results_df['punkty_score'] = round(points_predictions['Score'], 2)

results_df['bramki_dom'] = home_predictions['Label']
results_df['bramki_dom_score'] = round(home_predictions['Score'], 2)

results_df['bramki_wyjazd'] = away_predictions['Label']
results_df['bramki_wyjazd_score'] = round(away_predictions['Score'], 2)

inv_teams = {v: k for k, v in teams_dict.items()}

results_df['dom'] = results_df['dom'].replace(inv_teams)
results_df['wyjazd'] = results_df['wyjazd'].replace(inv_teams)

results_df['home_or_away'] = home_predictions['home_or_away']

results_df = results_df[results_df['home_or_away']==True]

results_df.drop(labels = ['home_or_away'], axis = 1, inplace = True)
results_df.reset_index(drop=True, inplace = True)

results_df['prob_sum'] = round((results_df.punkty_score + results_df.bramki_dom_score + results_df.bramki_wyjazd_score), 2)

results_df['prob_avg'] = round(((results_df.punkty_score + results_df.bramki_dom_score + results_df.bramki_wyjazd_score) / 3), 2)

results_df.sort_values('punkty_score', ascending=False, inplace = True)

results_df.reset_index(drop=True, inplace = True)

# SAVING RESULTS TO HTML

html_filename = 'from-' + str(start_d) + '-to-' + str(end_d) + '.html'

html_path = './data/outputs/'
html_local_path = 'file://' + os.path.realpath(html_path + html_filename)

html = results_df.to_html()

text_file = open(html_path + html_filename, 'w')
text_file.write(html)
text_file.close()

webbrowser.open(html_local_path)