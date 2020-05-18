# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 14:04:58 2020

@author: eSq
"""

import pandas as pd
from yahoofinancials import YahooFinancials
import datetime
import pandas_datareader.data as pdr
import pandas as pd


# close_prices = pd.DataFrame()
# end_date = (datetime.date.today()).strftime('%Y-%m-%d')
# beg_date = (datetime.date.today()-datetime.timedelta(1825)).strftime('%Y-%m-%d')
# cp_tickers = all_tickers
# attempt = 0
# drop = []
# while len(cp_tickers) != 0 and attempt <=5:
#     print("-----------------")
#     print("attempt number ",attempt)
#     print("-----------------")
#     cp_tickers = [j for j in cp_tickers if j not in drop]
#     for i in range(len(cp_tickers)):
#         try:
#             yahoo_financials = YahooFinancials(cp_tickers[i])
#             json_obj = yahoo_financials.get_historical_price_data(beg_date,end_date,"daily")
#             ohlv = json_obj[cp_tickers[i]]['prices']
#             temp = pd.DataFrame(ohlv)[["formatted_date","adjclose"]]
#             temp.set_index("formatted_date",inplace=True)
#             temp2 = temp[~temp.index.duplicated(keep='first')]
#             close_prices[cp_tickers[i]] = temp2["adjclose"]
#
#             print(cp_tickers[i])
#
#             drop.append(cp_tickers[i])
#         except:
#             print(cp_tickers[i]," :failed to fetch data...retrying")
#             continue
#     print(close_prices)
#     attempt+=1
#

def retrieve_financial_data_from_yahoo(symbols_list, start_date, end_date, output_directory=''):
    for symbol in symbols_list:
        data = pdr.DataReader(symbol,
                              start=start_date,
                              end=end_date,
                              data_source='yahoo')
        filepath = output_directory + symbol + '.csv'
        data.to_csv(path_or_buf=filepath)
        print('downloaded ' + symbol)

if __name__ == "__main__":
    all_tickers = ["NKE", "IBM", "AAPL", "EBAY", "INTC"]
    retrieve_financial_data_from_yahoo(all_tickers, '2000-01-02', '2020-01-02', 'data/')