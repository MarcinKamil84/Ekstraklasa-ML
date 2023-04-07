import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from yellowbrick.classifier import confusion_matrix
from yellowbrick.classifier.rocauc import roc_auc
from pycaret.classification import *


# IMPORTING DATA

ekstraklasa_balanced_train = pd.read_pickle('./data/inputs/ekstraklasa_train.pkl')

ekstraklasa_balanced_train = ekstraklasa_balanced_train.loc[:,~ekstraklasa_balanced_train.T.duplicated(keep='first')]


# PREPARING DATA

X_pycaret_points = ekstraklasa_balanced_train.drop(columns=['goals', 'goals_lost', 'team', 'opponent'])
X_pycaret_goals = ekstraklasa_balanced_train.drop(columns=['points', 'goals_lost', 'team', 'opponent'])

# PYCARET settings

categorical_features = ['home_or_away']
feature_selection_threshold = float((np.sqrt(len(X_pycaret_goals.columns))) / len(X_pycaret_goals.columns))


# # POINTS

# MODEL SELECTION
print("Comparing models...")

points = setup(
    data = X_pycaret_points,
    target = 'points',
    silent = True,
    polynomial_features = True,
    categorical_features = categorical_features,
    feature_selection = True,
    feature_selection_threshold = feature_selection_threshold
    )

best_points = compare_models(n_select = 1, sort = 'AUC')

results = pull(best_points)

# MODEL TRAINING
print("Model training...")

points_holdout = predict_model(best_points)

points_predictions = pull(best_points)

points_created = create_model(best_points)

roc_auc(points_created, get_config('X_train'), get_config('y_train'), X_test=get_config('X_test'), y_test=get_config('y_test'), classes=['2', 'x', '1'], show=False)
plt.savefig('./models/points_auc.png')
plt.clf()

confusion_matrix(points_created, get_config('X_train'), get_config('y_train'), X_test=get_config('X_test'), y_test=get_config('y_test'), show=False)
plt.savefig('./models/points_conf.png')
plt.clf()


# MODEL TUNING
print("Model tuning...")

points_tuned = tune_model(points_created, optimize='AUC', choose_better=True, n_iter = 50)

points_final = finalize_model(points_tuned)

points_all = predict_model(points_final)


# SAVING MODEL
print("Saving model...")
save_model(points_final, './models/points_final')


# # GOALS

# MODEL SELECTION
print("Comparing models...")


goals = setup(
    data = X_pycaret_goals,
    target = 'goals',
    silent = True,
    polynomial_features = True,
    categorical_features = categorical_features,
    feature_selection = True,
    feature_selection_threshold = feature_selection_threshold
    )

best_goals = compare_models(n_select = 1, sort = 'AUC')


# MODEL TRAINING
print("Model training...")

goals_holdout = predict_model(best_goals)

goals_predictions = pull(best_goals)

goals_created = create_model(best_goals)

roc_auc(goals_created, get_config('X_train'), get_config('y_train'), X_test=get_config('X_test'), y_test=get_config('y_test'), show=False)
plt.savefig('./data/img/goals_auc.png')
plt.clf()

confusion_matrix(goals_created, get_config('X_train'), get_config('y_train'), X_test=get_config('X_test'), y_test=get_config('y_test'), show=False)
plt.savefig('./data/img/goals_conf.png')
plt.clf()


# MODEL TUNING
print("Model tuning...")

goals_tuned = tune_model(goals_created, optimize='AUC', choose_better=True, n_iter = 50)

goals_final = finalize_model(goals_tuned)

goals_all = predict_model(goals_final)


# SAVING MODEL
print("Saving model...")
save_model(goals_final, './models/goals_final')

points_predictions.to_csv('./data/outputs/preds/points_predictions.csv')
goals_predictions.to_csv('./data/outputs/preds/goals_predictions.csv')
