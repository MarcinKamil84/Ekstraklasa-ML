print("<<< START >>>")

# IMPORTS

print("Importing modules...")

from pycaret.classification import *


# # Setting date span

print("Chose games dates to predict (yyyy-mm-dd):")

start_d_input = input("Start date: ")
end_d_input = input("End date: ")

start_d_file = open("data/variables/start_d.txt", 'w')
start_d_file.write(start_d_input)
start_d_file.close()

end_d_file = open("data/variables/end_d.txt", 'w')
end_d_file.write(end_d_input)
end_d_file.close()


# # CONFIGURATION QUESTIONS

question1 = input("Crawl for updated games? (y/n): ")
question3 = input("Crawl for recent news articles? (y/n): ")
question5 = input("Train new NLP model? (y/n): ")
question4 = input("Compile new features? (y/n): ")
question2 = input("Train new results model? (y/n): ")


# # CRAWLER

if question1 == "y":
    
    print("Crawling for updates...")
    
    import scripts.crawler as crawler

    crawler.run_spider()

else:
    
    print("Updates skipped.")
        
    pass


# # NEWS NLP TRAINING

if question5 == "y":

    print("Executing news sentiment analysis predictions...")

    from nlp import news_training_gol24

else:
    
    print("Skipping news sentiment nlp training.")
        
    pass


# # NEWS NLP SCRAPING

if question3 == "y":

    print("Executing news sentiment analysis predictions...")

    from scripts import newscrawler_gol24
    from nlp import news_predicting_gol24

else:
    
    print("Skipping news sentiment analysis predictions.")
        
    pass


# FEATURE ENGINEERING

if question4 == "y":

    print("Feature engineering...")

    from scripts import feature_engineering

else:
    
    print("Feature engineering skipped.")
        
    pass


# MACHINE LEARNING

if question2 == "y":
    
    print("Training new model...")
    
    from scripts import model_training

else:
    
    print("Skipping new model training...")


# PREDICTIONS

print("Making predictions...")

from scripts import predict_models


# VISUALIZING REPORT

print("Making predictions...")

from scripts import visualize

print("<<< END >>>")
