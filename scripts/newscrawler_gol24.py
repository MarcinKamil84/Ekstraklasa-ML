import pandas as pd
import re
from re import sub
from datetime import datetime
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from bs4 import BeautifulSoup
from crochet import setup, wait_for

setup()


# Checking last date in the existing json file

news_path = './data/news/articles_gol24.json'
news_df = pd.read_json(news_path)
date_max = news_df['date'].max().date()
#date_max = datetime.strptime('2000-01-01', '%Y-%m-%d').date()


# Getting a list of already visited urls

urls_file = pd.read_csv(r'./data/news/urls_gol24.csv')


# Create the Spider class
class News(scrapy.Spider):
    name = "news_spider"
    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
        'COOKIES_ENABLED': True,
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_TIMEOUT': 300,
        'DUPEFILTER_CLASS': 'scrapy.dupefilters.BaseDupeFilter'
    }
    
    # start_requests method
    
    def start_requests(self):

        base_url = "https://gol24.pl/ekstraklasa/"

        print("Starting scrapy...")

        yield scrapy.Request(url = base_url,
                         callback = self.parse_pages)
        
        
    # Second parsing method - parse by page
    
    def parse_pages(self, response):
        
        link_extractor = LinkExtractor(allow=["gol24", "/ar/"],
                                       restrict_xpaths='//div[contains(@class, "medium")] /a',
                                       unique=True)
        
        pages = link_extractor.extract_links(response)

        page_counter = 0

        for page in pages:

            if urls_file['urls'].eq(page.url).any():

                print('Page already scrapped.')

                continue

            else:

                page_counter += 1
                print('Scraping new page...')

                urls_file.loc[len(urls_file), 'urls'] = page.url

                yield response.follow(url = page,
                                    callback = self.parse_articles)

        urls_file.drop_duplicates(inplace=True, subset='urls', ignore_index=True)
        urls_file['urls'].to_csv(r'./data/news/urls_gol24.csv', mode='w')

        # Skip to next page

        next_link_extractor = LinkExtractor(allow=["gol24", 'ekstraklasa/'],
                                       restrict_xpaths='//a[contains(@rel, "next")]',
                                       unique=True)
        
        next_page_list = next_link_extractor.extract_links(response)
        next_page = next_page_list[0]


        if next_page: # If next page button exists

            if page_counter > 0: # and some new pages were scrapped
            
                yield response.follow(url = next_page,
                                      callback=self.parse_pages) # go to the next page
                
                page_counter = 0

            else:

                print('No new page to scrap.')


    # Third parsing method - parse actual articles
    
    def parse_articles(self, response):

        soup = BeautifulSoup(response.text, 'lxml')

        try:
            
            # Get date
            art_date = soup.find('time')
            art_date_class = art_date['datetime']

            # Get tags
            art_tags = soup.find('meta', {'name': 'keywords'})
            art_tags_class = art_tags['content']

            # Get title
            art_title = soup.find('h1', {'class' : 'atomsArticleHead__title'}).get_text()

            # Get content
            art_content = soup.find_all('div', {'class' : 'md'})
            
            phrases = []
            phrases2 = ""

            for phrase in art_content:
                phrase_ok = phrase.get_text()
                phrases.append(phrase_ok)
                phrases2 = ' '.join(phrases)

        except TypeError:

            # Ignore empty dates
            print("WARNING: TypeError occured.")

        else:
            
            if len(phrases2) != 0:

                # Building dataframe for export
 
                date_dict = str(re.findall(r"\d{4}\-\d{2}\-\d{2}", art_date_class)[0])
                date_object = datetime.strptime(date_dict, '%Y-%m-%d').date()

                if date_object > date_max:

                    articles_dict = {
                    
                    'date' : str(re.findall(r"\d{4}\-\d{2}\-\d{2}", art_date_class)[0]),
                    'tags' : str(art_tags_class),
                    'title' : str(art_title),
                    'length' : len(phrases2),
                    'content' : str(phrases2),

                    }   

                    articles_table.loc[len(articles_table)] = articles_dict


# Creating empty dataframes

articles_df_cols = ['date', 'tags', 'length', 'title', 'content']
articles_table = pd.DataFrame(columns=articles_df_cols)

# Running the Spider

@wait_for(30000)
def run_spider():
    process = CrawlerProcess()
    proc = process.crawl(News)
    return proc

run_spider()


# Formatting the data to export

articles_table.dropna(axis=0, how='any', subset=['date', 'content'], inplace=True)

articles_table['tags'] = articles_table['tags'].astype('string')
articles_table['title'] = articles_table['title'].astype('string')
articles_table['content'] = articles_table['content'].astype('string')


# Preparing list of words for nlp

def text_to_words(text):

    text = sub(r"\+", " plus ", text)
    text = sub(r",", " ", text)
    text = sub(r"\.", " ", text)
    text = sub(r"!", " ! ", text)
    text = sub(r"\?", " ? ", text)
    text = sub(r"'", " ", text)
    text = sub(r":", " : ", text)
    text = sub(r"\s{3,}", " ", text)
    text = sub(r"\d", " ", text)

    text = text.split()

    return text


articles_table['content_listed'] = articles_table['content'].apply(lambda x: text_to_words(x))


# Exporting the data original

'''
articles_table.to_json('./data/news/articles_gol24.json',
                       force_ascii = False,
                       orient="columns")
'''

# Exporting the data new

articles_table.to_json('./data/news/articles_gol24_update.json',
                       force_ascii = False,
                       orient="columns")