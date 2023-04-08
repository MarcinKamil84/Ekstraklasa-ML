# Ekstraklasa-ML

## Predicting Polish football league matches 

This project was made as an exercise in:

- data collection via scraping,
- NLP and sentiment analysis,
- feature engineering
- ML model training
- classification ML predictions

Project contents:

notebook.py : main app file to handle whole process

/scripts/
- crawler.py : historical results data collection
- news_training_gol24.py : NLP supervised training for sentiment analysis
- newscrawler_gol24 : sport news data collection
- news_predicting_gol24 : sentiment analysis for football news
- feature_engineering.py : data preprocessing and feature engineering
- model_training.py : model creation and training
- predict_models.py : making predictions based on the created models
- visualize.py : visualizing predictions

/data/
- contains files created in the process

/models/ [not included due to size limitations]
- contains models stored in pkl files during model training phase

/nlp/
- pretrained NLP models and datasets

Dependencies are included in requirements.txt file.

Additional resources for NLP data collection:
https://ermlab.com/en/blog/nlp/polish-sentiment-analysis-using-keras-and-word2vec/
http://dsmodels.nlp.ipipan.waw.pl/w2v.html
