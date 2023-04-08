import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from yellowbrick.classifier import confusion_matrix
from yellowbrick.classifier.rocauc import roc_auc
from pycaret.classification import *


# IMPORTING DATA

ekstraklasa_balanced_goals_train = pd.read_pickle('./data/inputs/ekstraklasa_balanced_goals_train.pkl')
ekstraklasa_balanced_goals_train = ekstraklasa_balanced_goals_train.loc[:,~ekstraklasa_balanced_goals_train.T.duplicated(keep='first')]

ekstraklasa_balanced_points_train = pd.read_pickle('./data/inputs/ekstraklasa_balanced_points_train.pkl')
ekstraklasa_balanced_points_train = ekstraklasa_balanced_points_train.loc[:,~ekstraklasa_balanced_points_train.T.duplicated(keep='first')]


# PREPARING DATA

X_pycaret_points = ekstraklasa_balanced_points_train.drop(columns=['goals', 'goals_lost', 'team', 'opponent'])
X_pycaret_goals = ekstraklasa_balanced_goals_train.drop(columns=['points', 'goals_lost', 'team', 'opponent'])

# PYCARET settings

categorical_features = ['home_or_away']
feature_selection_threshold = float((np.sqrt(len(X_pycaret_goals.columns))) / len(X_pycaret_goals.columns))


# # POINTS

# MODEL SELECTION
print("Comparing models...")

points = setup(
    data = X_pycaret_points,
    target = 'points',
    train_size = 0.7,
    data_split_stratify = True,
    silent = True,
    categorical_features = categorical_features,
    feature_selection = True,
    feature_selection_threshold = feature_selection_threshold
    )

best_points = compare_models(n_select = 1, sort = 'Accuracy')


# MODEL TRAINING
print("Model training...")

points_holdout = predict_model(best_points, verbose=False)

points_created = create_model(best_points, fold=10, verbose=False)

roc_auc(points_created, get_config('X_train'), get_config('y_train'), X_test=get_config('X_test'), y_test=get_config('y_test'), classes=['2', 'x', '1'], show=False)
plt.savefig('./models/points_auc.png')
plt.clf()

confusion_matrix(points_created, get_config('X_train'), get_config('y_train'), X_test=get_config('X_test'), y_test=get_config('y_test'), classes=['2', 'x', '1'], show=False)
plt.savefig('./models/points_conf.png')
plt.clf()

# MODEL TUNING
print("Model tuning...")

points_tuned = tune_model(points_created, optimize='AUC', choose_better=True, n_iter = 20, verbose=False)

points_ensemble = ensemble_model(points_tuned, fold = 10, method='Bagging', n_estimators = 20, choose_better = True, verbose=False)

points_final = finalize_model(points_ensemble)

roc_auc(points_final, get_config('X_train'), get_config('y_train'), X_test=get_config('X_test'), y_test=get_config('y_test'), classes=['2', 'x', '1'], show=False)
plt.savefig('./models/points_final_auc.png')
plt.clf()

confusion_matrix(points_final, get_config('X_train'), get_config('y_train'), X_test=get_config('X_test'), y_test=get_config('y_test'), classes=['2', 'x', '1'], show=False)
plt.savefig('./models/points_final_conf.png')
plt.clf()

points_all = predict_model(points_final)

points_predictions = pull()


# SAVING MODEL
print("Saving model...")
save_model(points_final, './models/points_final')


# # GOALS

# MODEL SELECTION
print("Comparing models...")


goals = setup(
    data = X_pycaret_goals,
    target = 'goals',
    train_size = 0.7,
    data_split_stratify = True,
    silent = True,
    categorical_features = categorical_features,
    feature_selection = True,
    feature_selection_threshold = feature_selection_threshold
    )


best_goals = compare_models(n_select = 1, sort = 'Accuracy')


# MODEL TRAINING
print("Model training...")

goals_holdout = predict_model(best_goals, verbose=False)

goals_created = create_model(best_goals, fold=10, verbose=False)

roc_auc(goals_created, get_config('X_train'), get_config('y_train'), X_test=get_config('X_test'), y_test=get_config('y_test'), classes=['2', 'x', '1'], show=False)
plt.savefig('./models/goals_auc.png')
plt.clf()

confusion_matrix(goals_created, get_config('X_train'), get_config('y_train'), X_test=get_config('X_test'), y_test=get_config('y_test'), classes=['2', 'x', '1'], show=False)
plt.savefig('./models/goals_conf.png')
plt.clf()

# MODEL TUNING
print("Model tuning...")

goals_tuned = tune_model(goals_created, optimize='AUC', choose_better=True, n_iter = 20, verbose=False)

goals_ensemble = ensemble_model(goals_tuned, fold = 10, method='Bagging', n_estimators = 20, choose_better = True, verbose=False)

goals_final = finalize_model(goals_ensemble)

roc_auc(goals_final, get_config('X_train'), get_config('y_train'), X_test=get_config('X_test'), y_test=get_config('y_test'), show=False)
plt.savefig('./models/goals_final_auc.png')
plt.clf()

confusion_matrix(goals_final, get_config('X_train'), get_config('y_train'), X_test=get_config('X_test'), y_test=get_config('y_test'), show=False)
plt.savefig('./models/goals_final_conf.png')
plt.clf()

goals_all = predict_model(goals_final)

goals_predictions = pull()


# SAVING MODEL
print("Saving model...")
save_model(goals_final, './models/goals_final')

points_predictions.to_csv('./data/outputs/preds/points_predictions.csv')
goals_predictions.to_csv('./data/outputs/preds/goals_predictions.csv')
