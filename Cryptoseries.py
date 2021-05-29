from bs4 import BeautifulSoup
import requests
import pandas as pd
import json
import time

print('b')

cmc= requests.get('https://coinmarketcap.com/')
soup = BeautifulSoup(cmc.content, 'html.parser')


#print(soup.prettify())
data = soup.find('script', id = "__NEXT_DATA__", type = "application/json")
coins = {}
coin_data = json.loads(data.contents[0])
listings = coin_data['props']['initialState']['cryptocurrency']['listingLatest']['data']

for i in listings:
	coins[str(i['id'])] = i['slug']

#for i in coins:
page = requests.get(f'https://coinmarketcap.com/currencies/{coins[i]}/historical-data/?start=20200101&end=20200630')
soup = BeautifulSoup(page.content, 'html.parser')
data = soup.find('script', id = "__NEXT_DATA__", type = "application/json")
	
historical_data = json.loads(data.contents[0])


quotes = historical_data['props']['initialState']['cryptocurrency']['ohlcvHistorical'][i]['quotes']
info = historical_data['props']['initialState']['cryptocurrency']['ohlcvHistorical'][i]

market_cap = []
volume = []
timestamp = []
name = []
symbol= []
slug = []

print('a')