import pandas as pd
import numpy as np
import dominate
from dominate.tags import *
import webbrowser
import os

# Loading data

res = pd.read_pickle('./data/outputs/preds/results_final.pkl')

res['wynik'].replace([0, 1, 2], ['2', 'x', '1'], inplace=True)

res['av_score'] = round((res['bramki_dom_score'] + res['bramki_wyjazd_score']) / 2, 2)

res[':'] = ':'

start_d_file = open("./data/variables/start_d.txt", 'r')
start_d_text = start_d_file.read()
start_d_file.close()

end_d_file = open("./data/variables/end_d.txt", 'r')
end_d_text = end_d_file.read()
end_d_file.close()

start_d = np.datetime64(str(start_d_text))
end_d = np.datetime64(str(end_d_text))

points_predictions = pd.read_csv('./data/outputs/preds/points_predictions.csv', header=None,  index_col=False)
points_predictions = points_predictions.iloc[:,1:]
goals_predictions = pd.read_csv('./data/outputs/preds/goals_predictions.csv', header=None, index_col=False)
goals_predictions = goals_predictions.iloc[:,1:]

# HTML variables

html_title = 'Ekstraklasa: predictions'
html_subtitle = 'from ' + str(start_d) + ' to ' + str(end_d)
html_table_columns_left = ['', '', '', 'prediction score']
html_table_cells_left = res[['dom', 'wynik', 'wyjazd', 'wynik_score']]
html_table_columns_right = ['', '', ':', '', '', 'prediction score home', 'prediction score away', 'prediction score avg.']
html_table_cells_right = res[['dom', 'bramki_dom', ':', 'bramki_wyjazd', 'wyjazd', 'bramki_dom_score', 'bramki_wyjazd_score', 'av_score']].sort_values('av_score', ascending=False)

# Path variables

html_filename = 'from-' + str(start_d) + '-to-' + str(end_d) + '.html'
html_path = 'data/outputs/html/'
html_local_path = os.path.realpath(html_path + html_filename)


# Create doc

doc = dominate.document(title=html_title)

with doc.head:
    link(rel='stylesheet', href='style.css')

with doc:

    h1('Ekstraklasa')
    h2('Predictions {}'.format(html_subtitle))

    with div(cls='left_div'):

        h3('Who wins?')

        with table(cls='wyniki'):
            with tr():
                for i in html_table_columns_left:
                    td(i)
            
            for j, row in html_table_cells_left.iterrows():
                with tr():
                    for cell in html_table_cells_left.columns:
                        td_cell = td(html_table_cells_left.loc[j, cell])
                        td_color = 'background-color: rgb(0, 170, 23, {})'.format(html_table_cells_left.loc[j, 'wynik_score'])
                        td_cell['style'] = td_color
                        print(td_cell)

    with div(cls='left_div'):

        h3('Goals')

        with table(cls='gole'):
            with tr():
                for i in html_table_columns_right:
                    td(i)
            
            for j, row in html_table_cells_right.iterrows():
                with tr():
                    for cell in html_table_cells_right.columns:
                        td_cell = td(html_table_cells_right.loc[j, cell])
                        td_color = 'background-color: rgba(207, 117, 0, {})'.format(html_table_cells_right.loc[j, 'av_score'])
                        td_cell['style'] = td_color                        
                        print(td_cell)

    with div(cls='left_div clears'):

        h3('ML model details')

    with div(cls='left_div'):

        p("Model for results:")

        with table(cls='modele'):
            for j, row in points_predictions.iterrows():
                with tr():
                    for cell in points_predictions.columns:
                        td_cell = td(points_predictions.loc[j, cell])
                        td_color = 'background-color: rgba(0, 73, 230, {})'.format(points_predictions.loc[j, cell])
                        td_cell['style'] = td_color                        
                        print(td_cell)

        img(src='..\..\img\points_final_auc.png')
        img(src='..\..\img\points_final_conf.png')

    with div(cls='left_div'):

        p("Model for goals:")

        with table(cls='modele'):
            for j, row in goals_predictions.iterrows():
                with tr():
                    for cell in goals_predictions.columns:
                        td_cell = td(goals_predictions.loc[j, cell])
                        td_color = 'background-color: rgba(0, 73, 230, {})'.format(goals_predictions.loc[j, cell])
                        td_cell['style'] = td_color                        
                        print(td_cell)

        img(src='..\..\img\goals_final_auc.png')
        img(src='..\..\img\goals_final_conf.png')


# Opening summary

htmlsave = open(html_path + html_filename, 'w+')
htmlsave.write(str(doc))
htmlsave.close()

webbrowser.open(html_local_path)
