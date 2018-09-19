# from urllib import urlopen
import urllib2
import urlparse
import time
import pprint
from BeautifulSoup import BeautifulSoup
url = 'http://www.answers.com/Q/FAQ/3408'
page = urllib2.urlopen(url)
soup = BeautifulSoup(page)
categories = {}
for link in soup.findAll(attrs={'class':"category_name"}):
    category_name = link.text
    link = link.find('a')
    category_link = link.get('href')
    categories[category_name] = category_link

    file_name = category_name+'.txt'
    file = open(file_name,'a')
    inputURL = category_link
    for i in range(1,21):
        URL = inputURL+'-'+str(i)
        html_page = urllib2.urlopen(URL)
        soup = BeautifulSoup(html_page)
        questions = soup.findAll(attrs={'class':"question"})
        for link in questions:
            file.write(link.text.encode('utf8')+'\n')
    file.close()
