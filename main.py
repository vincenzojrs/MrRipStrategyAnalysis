#
import csv
import polars as pl
import yfinance as yf

class Data_Engineering():
    def __init__(self):
        self.value_asset_data = dict()
        self.price_data = dict()
        self.amount_asset_data = dict()

    def value_asset_data_retriever(self, path_file, asset_line_end, asset_line_start = 3):
        
        with open(path_file,   mode  = 'r') as f: # Apri il file; utilizza csv.reader perché gestisce i decimali con la virgola
            reader = csv.reader(f)
            
            for _ in range(asset_line_start): # Skippa le righe fino ad "asset_line_start" nel file excel
                next(reader)

            for i, values in enumerate(reader): # Assegna al dizionario asset_data il primo valore di ogni riga e quelli relativi ad ogni mese
                if i == asset_line_end - asset_line_start: # Considera tutte le righe entro "asset line end"
                    break
                self.value_asset_data[values[0]] = list(map(lambda x : int(x.replace(',', '')), values[3:15])) # Assegna a ogni ticker una lista che contiene il valore degli asset. Ad ogni valore è rimossa la virgola e "int"-erizzata

    def price_data_retriever(self, year):
        tickers = list(self.value_asset_data.keys()) # Ottieni la lista di tickers dal file degli asset
        for ticker in tickers: # Per ogni ticker, ottieni i dati giornalieri di chiusura del prezzo, quindi campiona i dati per ottenere i prezzi di chiusura degli ultimi giorni
            self.price_data[ticker] = yf.Ticker('VT').history(start = str(year)+'-01-01', end = str(year+1)+'-01-01')['Close'].resample('ME').last().tolist()

    def amount_data_calculator(self):
        for key in self.value_asset_data:
            self.amount_asset_data[key] = [value / price for value, price in zip(self.value_asset_data[key], self.price_data[key])]
        
data_2023 = Data_Engineering()
data_2023.value_asset_data_retriever('source_files/RIP Net Worth (mr.rip_nw) - 2023.csv', 13)
data_2023.price_data_retriever(2023)
data_2023.amount_data_calculator()
