import csv
import polars as pl
import yfinance as yf
import hvplot

class Data_Engineering():
    def __init__(self, year):
        self.year = year
        self.value_asset_data = dict()
        self.price_data = dict()
        self.amount_data = dict()

    @staticmethod
    def to_polars(dictionary):
        """
        To convert dictionaries into polars dataframe
        """
        return pl.DataFrame(dictionary) 
    
    @staticmethod
    def _concatenate_dicts(dict1, dict2):
        """
        Hidden function which concatenates two dictionaries based on common keys, values for each key are already sorted

        Args:
        - dict1 (dict): First dictionary.
        - dict2 (dict): Second dictionary.

        Returns:
        - concatenated_dict (dict): A new dictionary with concatenated lists for matching keys.
        """

        concatenated_dict = {}
        keys = set(dict1.keys()) & set(dict2.keys()) # find the intersection of keys
        for key in keys: # for each commons keys, concatenate their values
            concatenated_dict[key] = dict1[key] + dict2[key]
        return concatenated_dict

    def value_asset_data_retriever(self, asset_line_end, asset_line_start=3, to_polars=False):
        """
        Retrieve data from a file stored in the source_files folder, whose name is in the format "YYYY.csv". Lines contains monthly value for each asset in rows, calculated at the end of each month.
        
        Args:
            asset_line_end (int): what's the last line corresponding to an asset
            asset_line_start (int, default = 3): what's the first line corresponding to an asset
            to_polars (bool, default = False): convert to polar

        """
        with open('source_files/' + str(self.year) + '.csv', mode='r') as f:
            reader = csv.reader(f)
            # Skip the first 3 lines
            for _ in range(asset_line_start):
                next(reader)

            # create a dictionary whose keys are tickers and values are the list of values in dollars for each asset, for each month
            for i, values in enumerate(reader):
                if i == asset_line_end - asset_line_start:
                    break
                self.value_asset_data[values[0]] = list(map(lambda x: int(x.replace(',', '')), values[3:15]))

        if to_polars:
            self.value_asset_data = self.to_polars(self.value_asset_data)

    def price_data_retriever(self, sampling = 'ME', to_polars=False):
        """
        Retrieve closing price data from yahoo finance for each ticker, in the given year
        
        Args:
            sampling (string, default = 'ME'): what's the frequency of data? 'ME' for monthly data
            to_polars (bool, default = False): convert to polar

        """
        tickers = list(self.value_asset_data.keys())
        for ticker in tickers:
            if ticker == "INTEL":   # to be deleted
                continue            # to be deleted
            self.price_data[ticker] = yf.Ticker(ticker).history(start = str(self.year) + '-01-01', 
                                                                end=str(self.year + 1) + '-01-01')['Close']
            if sampling == 'ME':
                self.price_data[ticker] = self.price_data[ticker].resample('ME').last().tolist()
                
        if to_polars:
            self.price_data = self.to_polars(self.price_data)

    def amount_data_calculator(self, to_polars=False):
        """
        Calculate the amount of asset each month, per each ticker
        """

        for key in self.value_asset_data:
            if key == "INTEL":      # to be deleted
                continue            # to be deleted
            self.amount_data[key] = [value / price for value, price in zip(self.value_asset_data[key], self.price_data[key])]

        if to_polars:
            self.amount_data = self.to_polars(self.amount_data)

    def get_difference(self, mode, diff=1, pct_change=False):
        """
        Calculate absolute or percentuale change in price or amount of asset per each moment
        
        Args:
            mode (string): "price" or "amount", according to the data to calculate the difference of
            diff (int, default = 1): lag
            pct_change (bool, default = False): calculate the percentuale change
        
        Returns:
            pl.Dataframe (polar dataframe)
        """
        if mode == "price":
            data = self.price_data if isinstance(self.price_data, pl.DataFrame) else self.to_polars(self.price_data)
            if pct_change:
                self.price_difference_pct = data.select(pl.all().pct_change(diff))
                return self.price_difference_pct
            else:
                self.price_difference = data.select(pl.all().diff(diff))
                return self.price_difference

        elif mode == "amount":
            data = self.amount_data if isinstance(self.amount_data, pl.DataFrame) else self.to_polars(self.amount_data)
            self.difference_amount = data.select(pl.all().diff(diff))
            return self.difference_amount

    def concatenate_data(self, older):
        """
        Concatenate data from another DataEngineering instance into this one.

        Args:
        - older (InstanceClass): Another instance of DataEngineering to concatenate with.

        Returns:
        - new_instance (InstanceClass): A new instance with concatenated data.
        """
        concatenated_value_asset_data = self._concatenate_dicts(older.value_asset_data, self.value_asset_data)
        concatenated_price_data = self._concatenate_dicts(older.price_data, self.price_data)
        concatenated_amount_data = self._concatenate_dicts(older.amount_data, self.amount_data)
        
        # Create a new instance with default year and updated data
        new_instance = Data_Engineering(year = self.year)
        new_instance.value_asset_data = concatenated_value_asset_data
        new_instance.price_data = concatenated_price_data
        new_instance.amount_data = concatenated_amount_data
        return new_instance

data_2023 = Data_Engineering(2023)
data_2023.value_asset_data_retriever(13)
data_2023.price_data_retriever()
data_2023.amount_data_calculator()

data_2022 = Data_Engineering(2022)
data_2022.value_asset_data_retriever(16)
data_2022.price_data_retriever()
data_2022.amount_data_calculator()

# Concatenate data

data_2022_2023 = data_2023.concatenate_data(data_2022)
test = data_2022_2023.get_difference(mode = 'amount', pct_change = False)

# %%
print(test)

# %%
test.plot()

# %%



