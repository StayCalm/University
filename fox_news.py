import re
import requests
import datetime
from bs4 import BeautifulSoup
from lxml import etree as ET
import datetime
import json
from googletrans import Translator
from inscriptis import get_text
from inscriptis.css import DEFAULT_CSS, HtmlElement

#https://www.foxnews.com/api/article-search?isCategory=true&isTag=false&isKeyword=false&isFixed=false&isFeedUrl=false&searchSelected=world&contentTypes=%7B%22interactive%22:true,%22slideshow%22:true,%22video%22:false,%22article%22:true%7D&size=11&offset=20
now = datetime.datetime.now()
j = 0
l = 0




translator = Translator()


url_science = 'https://www.foxnews.com/api/article-search?isCategory=true&isTag=false&isKeyword=false/' \
      '&isFixed=false&isFeedUrl=false&searchSelected=science&contentTypes=%7B%22interactive%22:true,\
      %22slideshow%22:false,%22video%22:false,%22article%22:true%7D&size=11&offset=7'
#url_2 = 'https://www.rbc.ua/rus/search?search_text=%D1%82%D1%80%D0%B0%D0%BC%D0%BF&submit=%D0%9F%D0%BE%D0%B8%D1%81%D0%BA&r=0&s=2&p=1'
#url_1 = 'https://tass.ru/userApi/mainPageNewsList'
'''https://www.foxnews.com/api/article-search?isCategory=true&isTag=false&isKeyword=false&isFixed=false\
&isFeedUrl=false&searchSelected=science\
&contentTypes=%7B%22interactive%22:true,%22slideshow%22:true,%22video%22:true,%22article%22:true%7D&size=11&offset=27'''
url_world = 'https://www.foxnews.com/api/article-search?isCategory=true&isTag=false&isKeyword=/' \
            'false&isFixed=false&isFeedUrl=false&searchSelected=world&contentTypes=%7B%22interactive%22:true,\
            %22slideshow%22:true,%22video%22:false,%22article%22:true%7D&size=11&offset=70'


def get_url_list(url):
      lst_urls = []
      response = requests.get(url)
      #print(response.text)
      if(response.text is not None or response.text is not ''):
            events_json = json.loads(response.text)
      else:
            print("Your json is None")
      for val in events_json:
            #print(val)
            if val is not None:
                  lst_urls.append(val['url'])
      return lst_urls



def get_event_content(new_url):
      try:
            response = requests.get(new_url)
            correct_url = True
      except:
            print("Please enter a valid URL")
      if correct_url is True:
            html = response.content.decode('utf-8')
            features = "lxml"
            soup = BeautifulSoup(html, features)
            ss = soup.find_all('p')
            '''text_of_new = re.sub(r'This material may not be published, broadcast, rewritten,\
            or redistributed\. ©2019 FOX News Network, LLC\. All rights reserved\.\
            All market data delayed 20 minutes\.','',BeautifulSoup(str(ss), "lxml").text)'''
            text_of_new = re.sub(r'^.{163}||.{163}$||CLICK HERE TO GET THE FOX NEWS APP', '',BeautifulSoup(str(ss), "lxml").text)
            return text_of_new

def create_files(f_name, text):
      f = open(f_name, "w+")
      f.write(text)
      f.close()
science = get_url_list(url_science)
j = 1
create_files( '/Users/amalakhova/Documents/news_site/parser/events/event/AAAWorldpoliticstrump-holds-joint-presser-with-turkish-president-erdogan_ru.txt', translator.translate(get_event_content('https://www.foxnews.com/politics/trump-holds-joint-presser-with-turkish-president-erdogan'),dest='ru').text)
#print(get_event_content('https://www.foxnews.com/world/how-al-baghdadi-isis-identified-dna'))
while(j <= 100 ):
      print(j)
      world = get_url_list('https://www.foxnews.com/api/article-search?isCategory=true&isTag=false&isKeyword=/' \
            'false&isFixed=false&isFeedUrl=false&searchSelected=world&contentTypes=%7B%22interactive%22:true,\
            %22slideshow%22:true,%22video%22:false,%22article%22:true%7D&size=11&offset='+ str(j))
      '''world = get_url_list('https://www.foxnews.com/api/article-search?isCategory=true&isTag=false&isKeyword=false/' \
      '&isFixed=false&isFeedUrl=false&searchSelected=science&contentTypes=%7B%22interactive%22:true,\
      %22slideshow%22:false,%22video%22:false,%22article%22:true%7D&size=11&offset=' + str(j))'''
      for i in world:
            print(i)
            print(translator.translate(get_event_content('https://www.foxnews.com' + str(i)), dest='ru').text)
            create_files( '/Users/amalakhova/Documents/news_site/parser/events/event/W'+re.sub(r'\/', '', i) + '_ru.txt', translator.translate(get_event_content('https://www.foxnews.com' + str(i)),dest='ru').text + 'https://www.foxnews.com' + str(i))
      j += 10


world = get_url_list(url_world)
#print(get_url_list(url_1))
#https://www.foxnews.com/opinion/baghdadi-dead-trump-future-james-carafano
#https://www.foxnews.com/media/washington-post-max-boot-al-baghdadi
#create_files( 'AAAWorld'+re.sub('\/', '', 'https://www.foxnews.com/media/washington-post-max-boot-al-baghdadi') + '_ru.txt', translator.translate(get_event_content('https://www.foxnews.com/world/isis-al-baghdadi-underwear-spy-dna-test-kurds'),dest='ru').text)
#for i in world:
      #create_files('World'+re.sub('\/', '', i) + '_ru.txt', get_event_content('https://www.foxnews.com/' + i).text)
      #create_files( 'World'+re.sub('\/', '', i) + '_ru.txt', translator.translate(get_event_content('https://www.foxnews.com/' + i),dest='ru').text)
