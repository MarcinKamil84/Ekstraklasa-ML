#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[3]:


import pandas as pd

import scrapy

from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor

from bs4 import BeautifulSoup

from crochet import setup, wait_for


# In[7]:


# Setting up the crawler

setup()

# Create the Spider class
class Ekstraklasa(scrapy.Spider):
    name = "ekstraklasa_spider"
    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
        'COOKIES_ENABLED': True,
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_TIMEOUT': 300,
        'DUPEFILTER_CLASS': 'scrapy.dupefilters.BaseDupeFilter'
    }
    
    # start_requests method
    
    def start_requests(self):
        base_url = 'https://www.worldfootball.net/all_matches/pol-ekstraklasa-2022-2023/'
        yield scrapy.Request(url = base_url,
                         callback = self.parse_seasons)
        
        
    # Second parsing method - parse by season
    
    def parse_seasons(self, response):
        
        options = response.xpath( '//table[@class="auswahlbox"]//option[contains(@value, "all_matches")]/@value' ).extract()
        
        # Using only data from 1989/90 seasons and newer
        
        for option in options:
          if "20" in option:
            yield response.follow(url = option,
                                  callback = self.parse_games)
          elif "199" in option:
            yield response.follow(url = option,
                                  callback = self.parse_games)
          else:
            continue
            
    # Third parsing method - parse actual results
    
    def parse_games(self, response):
        soup = BeautifulSoup(response.text, 'lxml')
        
        table = soup.find('table', {'class': 'standard_tabelle'})
        
        for row in table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) == 7:
                cell = [td.text.strip() for td in cells]
                teams_table.loc[len(teams_table)] = cell
        
        filename = './data/ekstraklasa.csv'
        teams_table.to_csv(filename, encoding='utf-8', mode='w')


# In[8]:


# Initialize the dictionary **outside** of the Spider class

df_cols = ['data', 'godzina', 'dom', '-', 'wyjazd', 'wynik', 'pusta']
teams_table = pd.DataFrame(columns=df_cols)

# Run the Spider

@wait_for(30)
def run_spider():
    process = CrawlerProcess()
    proc = process.crawl(Ekstraklasa)
    return proc


# In[9]:


# Print a preview of courses

print(teams_table)

