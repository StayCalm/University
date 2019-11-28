import re
import requests
import datetime
from bs4 import BeautifulSoup
from lxml import etree as ET
import datetime
import json


url_world = 'https://ria.ru/services/world/more.html?&date=20191026T150000'
def get_links(url):
    event_link = set()
    response = requests.get(url)
    html = response.content.decode('utf-8')
    features = "lxml"
    soup = BeautifulSoup(html, features)
    for link in soup.findAll('a'):
        event_link.add((link.get('href')))
    return event_link

def get_event_text(url):
    correct_url = False
    try:
        response = requests.get(url)
        correct_url = True
    except:
        print("Please enter a valid URL")
    if correct_url is True:
        html = response.content.decode('utf-8')
        features = "lxml"
        soup = BeautifulSoup(html, features)
        mydivs = soup.findAll("div", {"class": "article__block", "data-type": "text"})
        ty = BeautifulSoup(str(mydivs), "lxml").text
        return ty


def create_files(f_name, text):
    f = open(f_name, "w+")
    if text is not None:
        f.write(text)
    f.close()

j = 1
#https://ria.ru/services/science/more.html?id=1560679352&date=20191107T161858
while(j < 23):
    world = get_links('https://ria.ru/services/world/more.html?&date=20191123T' + str(j) +'0000')

    #world =  get_links('https://ria.ru/services/science/more.html?id=1560679352&date=20191107T' + str(j) +'0000')
    i = 0
    for val in world:
        i += 1
        print(i)
        create_files('/Users/amalakhova/Documents/news_site/parser/events/event/' + re.sub('\/', '', val) + '_ru.txt', get_event_text(val) + '\n'+str(val))
    j += 1
world = get_links(url_world)
#https://ria.ru/20191028/1560320905.html
#https://ria.ru/20191028/1560320729.html
world = ['https://ria.ru/20191028/1560320905.html']

#url1 = 'https://ria.ru/20191024/1560189972.html'
i = 0
'''for val in world:
    i += 1
    print(i)
    create_files('AAAWorld'+re.sub('\/', '', val) + '_ru.txt', get_event_text(val))'''