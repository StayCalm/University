import re
import requests
import datetime
from bs4 import BeautifulSoup
from lxml import etree as ET
import datetime
import json


#url = 'https://korrespondent.net/all/world/'
#url = 'https://korrespondent.net/all/world/2019/november/'
url = 'https://korrespondent.net/all/ukraine/2019/november/'
#url = 'https://korrespondent.net/all/world/russia/2019/november/'
def get_links(url):
    event_link = set()
    response = requests.get(url)
    html = response.content.decode('utf-8')
    features = "lxml"
    soup = BeautifulSoup(html, features)
    s = soup.findAll(class_='article__title')
    event_link  = re.findall('href="(.*?)">', str(s))
    return event_link


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
        text_of_new = re.sub(r'Новости от Корреспондент.net в Telegram. Подписывайтесь на наш канал https://t.me/korrespondentnet', '',
                             BeautifulSoup(str(ss), "lxml").text)
        return text_of_new

def create_files(f_name, text):
    f = open(f_name, "w+")
    if text is not None:
        f.write(text)
    f.close()


world = get_links(url)
j = 1
while(j < 40):
    world = get_links(url + 'p' + str(j))
    for i in world:
        create_files('/Users/amalakhova/Documents/news_site/parser/events/event/korr' + re.sub('\/', '', i) + '_ru.txt',get_event_content(i) + i)
    j += 1

print(get_links(url))