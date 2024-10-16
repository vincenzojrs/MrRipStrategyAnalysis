# %%
import csv
import polars as pl
import yfinance as yf
import pyarrow
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import linregress

plt.style.use('bmh')

# %%
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
                self.price_difference_pct = data.select([
                    pl.col(col).pct_change(diff).alias(f"{col}_{diff}_diff_pct_p") for col in data.columns
                ])
                return self.price_difference_pct
            else:
                self.price_difference = data.select([
                    pl.col(col).diff(diff).alias(f"{col}_{diff}_diff_p") for col in data.columns
                ])
                return self.price_difference

        elif mode == "amount":
            data = self.amount_data if isinstance(self.amount_data, pl.DataFrame) else self.to_polars(self.amount_data)
            self.difference_amount = data.select([
                pl.col(col).diff(diff).alias(f"{col}_{diff}_diff_q") for col in data.columns
            ])
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

# %%
prices = pd.DataFrame(data_2022_2023.price_data)

quantity = pd.DataFrame(data_2022_2023.amount_data)
quantity_change = data_2022_2023.get_difference("amount", diff = 1, pct_change= False).to_pandas()

value = pd.DataFrame(data_2022_2023.value_asset_data)

# %% [markdown]
# # Exploratory Data Analysis

# %% [markdown]
# ## Focus on Value

# %% [markdown]
# * All'inizio dell'analisi, a gennaio 2022, il portafoglio in oggetto raggiungeva un valore di circa 750'694€.
# * Il valore del portafoglio ha goduto di un trend positivo negli anni in oggetto, con un valore al termine dell'analisi di 939'886€; con una crescita mensile media del valore del portafogli di 10440€ e una variazione nell'intero periodo del 25.20%.
# * Il minimo globale è stato raggiunto nel giugno 2022, di circa 657'995€, con un drawdown del -12% dai massimi precedenti.
# * Lo stesso mese è stato raggiunto il massimo drawdown mensile dell'intero periodo, pari al -9.45%.
# * Un ulteriore importante drawdown è stato registrato ad ottobre 2023, pari al -8.77%.
# * L'oscillazione positiva massima è stata raggiunta a novembre 2022, pari al  +8.30%-
# * Il ritorno medio mensile è pari al +1.10%, mentre la deviazione standard dei rendimenti mensili è pari al 5.04%

# %%
def subplotting_dataframe(dataframe):
    """
        Plotting each column in a dataframe using subplots

        Parameters: 
        dataframe (pd.DataFrame)
    """
    num_columns = len(dataframe.columns)
    fig, axs = plt.subplots(num_columns, 1, figsize=(8, 2 * num_columns))
    plt.gcf().set_dpi(150)

    # Plot each column in a separate subplot
    for i, column in enumerate(dataframe.columns):
        axs[i].plot(dataframe.index, dataframe[column])
        axs[i].set_title(f'{column}')
        axs[i].grid(True)

    plt.axhline(y = 0, color = 'black', alpha = 0.5, linewidth = 1)
    plt.tight_layout()
    plt.show()

# %%
monthly_value = pd.DataFrame([value.sum(axis =1), value.sum(axis = 1).pct_change()*100]).T
monthly_value.rename(columns = {0:"abs", 1:"return"}, inplace = True)     

# %%
print(f"\n{'='*50}")
print(f"{'Analysis Report':^50}")
print(f"{'='*50}\n")

print(f"First period value: {monthly_value['abs'][0]:,}")
print(f"Last period value: {monthly_value['abs'][23]:,}")
print(f"Average monthly variability: {int(linregress(monthly_value.index, monthly_value['abs'])[0]):,}")
total_variation = (monthly_value['abs'][23] - monthly_value['abs'][0]) / monthly_value['abs'][0] * 100
print(f"Total variation: {total_variation:.2f}%")
print(f"Minimum of {monthly_value['abs'].min():,} reached at month {monthly_value['abs'].idxmin()+1}")
print(f"Maximum of {monthly_value['abs'].max():,} reached at month {monthly_value['abs'].idxmax()+1}")
print(f"Mean of Asset Value: {monthly_value['abs'].mean():,.2f}")
print(f"Standard Deviation of Asset Value: {monthly_value['abs'].std():,.2f}\n")

print(f"{'-'*50}")
print(f"{'Return Analysis':^50}")
print(f"{'-'*50}\n")

print(f"Minimum return: {monthly_value['return'].min():.2f}% reached at month {monthly_value['return'].idxmin()+1}")
print(f"Maximum return: {monthly_value['return'].max():.2f}% reached at month {monthly_value['return'].idxmax()+1}")
print(f"Mean of monthly returns: {monthly_value['return'].mean():.2f}%")
print(f"Standard Deviation of monthly returns: {monthly_value['return'].std():.2f}%\n")

# formatted using chatgpt :)

# %%
subplotting_dataframe(monthly_value)

# %% [markdown]
# ## Focus on Quantity

# %%
quantity.head()

# %%
quantity.plot(kind = 'line', figsize=(16,9), colormap = 'tab20b')
plt.gcf().set_dpi(300)
plt.xlabel('Month')
plt.ylabel('Number of Asset')
plt.title('Number of Asset per Month, 2022-2023')
plt.legend(title='Asset', loc='upper left')
plt.show()

# %% [markdown]
# ## Focus on purchasing strategy each month

# %%
quantity_change.head()

# %%
quantity_change.plot(kind = 'bar', figsize = (16,9), stacked = True, width = 0.9, colormap = 'tab20b')
plt.gcf().set_dpi(300)
plt.xlabel('Month')
plt.ylabel('Net number of asset traded')
plt.title('Net number of asset traded each month, 2022-2023')
plt.legend(title='Asset', loc='upper left')
plt.axhline(y = 0, color = 'black', alpha = 0.7)
plt.show()

# %% [markdown]
# # Detecting statistically significant correlations in asset purchasing
# Given:
# * a set of asset $A_{n}$,
# * their quantity bought each month $Quantity_{A, t}$,
# * their monthly price $Price_{A, t}$
# 
# For each asset $A$ is calculated the pearson correlation between its Quantity and Quantities and Prices of any other asset, at different time lags, as well.
# The variables significantly correlated will be returned, and their correlation coefficients.

# %%
def add_suffix_and_combine_dfs(dfs, suffixes):  
    """
    Combine dataframes and add suffixes in case they share the same columns
    Parameters:
    dfs (list of pd.DataFrame) : A list containing the dataframes to be concatenated
    suffixes (list of strings) : A list containing the suffixes to be added to each dataframe. REMEMBER: also '' can be used

    Returns:
    combined_df (pd.DataFrame) : The dataframe combined
    """
    modified_dfs = []
    
    for df, suffix in zip(dfs, suffixes):
        # Add suffix to column names
        df.columns = [f"{col}{suffix}" for col in df.columns]
        modified_dfs.append(df)
    
    # Combine DataFrames horizontally
    combined_df = pd.concat(modified_dfs, axis=1)
    
    return combined_df


# %%
def calculate_correlation(dataframe):
    """
    Calculate correlations among each pair of columns and display only the significant ones.
    
    Parameters:
    dataframe (pd.DataFrame): The DataFrame to calculate correlations for.
    
    Returns:
    dict: A dictionary with significant correlations.
    """
    
    columns = dataframe.columns
    correlations = {}
    
    for i, column_a in enumerate(columns):
        for column_b in columns[i + 1:]:  # Ensure we only compare each pair once
            # Check if both columns end with 'p'
            if column_a.endswith('p') and column_b.endswith('p'):
                continue
            
            # Calculate Pearson correlation and p-value
            corr, p_value = pearsonr(dataframe[column_a], dataframe[column_b])
            
            # Record significant correlations
            if p_value <= 0.05:
                correlations[f"{column_a} vs. {column_b}"] = float(corr)

    return correlations

# %%
price_change_1 = data_2022_2023.get_difference('price', diff = 1, pct_change = False).to_pandas()
price_change_2 = data_2022_2023.get_difference('price', diff = 2, pct_change = False).to_pandas()
price_change_3 = data_2022_2023.get_difference('price', diff = 3, pct_change = False).to_pandas()

correlation_dataframe = add_suffix_and_combine_dfs([quantity, quantity_change, prices, price_change_1, price_change_2, price_change_3], ['q', '', 'p', '', '', '']).dropna()

calculate_correlation(correlation_dataframe)

# %% [markdown]
# # Detecting purchasing strategy: focusing on the amount of stocks owned each month of the same asset using a time-series approach
# Given:
# * an the array of the number of units, for each month $A_{t}$,
# 
# For each asset $A$, using time-series analysis, discovering any statistically significant association between $Quantity_{A, t}$ at any given time and $Quantity_{A, t-n}$

# %%
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
warnings.simplefilter('ignore', InterpolationWarning)

# %%
def subplotting_dataframe(dataframe):
    """
        Plotting each column in a dataframe using subplots

        Parameters: 
        dataframe (pd.DataFrame)
    """
    num_columns = len(dataframe.columns)
    fig, axs = plt.subplots(num_columns, 1, figsize=(8, 2 * num_columns))

    # Plot each column in a separate subplot
    for i, column in enumerate(dataframe.columns):
        axs[i].plot(dataframe.index, dataframe[column])
        axs[i].set_title(f'{column}')
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()

# %%
def adf_test(dataframe):
    """
        Calculating stationarity on each dataframe column using KPSS and ADF tests.
        Args:
            dataframe (pandas.Dataframe) : dataframe whose columns are time series and stationarity should be calculated
    """
    for column in dataframe.columns:
        adf_test = adfuller(dataframe[column], autolag="AIC")
        kpss_test = kpss(dataframe[column], regression="c", nlags="auto")
        print("Stationarity check for", column)
        if adf_test[1] < 0.05:
            print("The series is stationary based on ADF test")
        else: 
            print("The series is not stationary based on ADF test")
        if float(kpss_test[1]) < 0.05:
            print("The series is not stationary based on KPSS test\n")
        else:
            print("The series is stationary based on KPSS test\n")

# %%
subplotting_dataframe(quantity)

# %%
adf_test(quantity)

# %% [markdown]
# # Detecting purchasing strategy: focusing on the amount bought each month of the same asset using a time-series approach
# Given:
# * a set of asset $A_{n}$,
# * their quantity bought each month $Quantity_{A, t}$,
# 
# For each asset $A$, using time-series analysis, discovering any statistically significant association between $Quantity_{A, t}$ at any given time and $Quantity_{A, t-n}$

# %% [markdown]
# Interestingly enough, the couples
# * VT - VYM
# * CHECK - ERUS
# * VBR - SONY
# * VYMI - VSS 
# 
# show some visual similarities. However, correlation among different asset will be assessed in a different section

# %%
subplotting_dataframe(quantity_change)

# %%
adf_test(quantity_change.dropna())

# %% [markdown]
# Plotting Time Series, ACF and PACF

# %%
def plotting(function):
    x_values = np.arange(0, len(function))
    plt.plot(x_values, function)
    plt.axhline(0, color='grey')
    plt.axhline(confidence_interval, linestyle='dashed')
    plt.axhline(-confidence_interval, linestyle='dashed')
    plt.xticks(x_values)

for column in amount_data.columns:
    # For each amount of asset bought and sold, calculate the auto correlation and partial autocorrelation functions
    acf_values = acf(amount_data[column])
    pacf_values = pacf(amount_data[column])

    # Define a confidence interval
    confidence_interval = 1.96 / np.sqrt(len(amount_data[column]))
    
    # If any of ACF-PACF value exceed the confidence interval excluding the first one - which is always statistically significant - , display the time series, the ACF and PACF
    if any(np.abs(acf_values[1:]) > confidence_interval) or any(np.abs(pacf_values[1:]) > confidence_interval):
        
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])

        # Time series plot spanning two rows
        ax0 = fig.add_subplot(gs[:, 0]) 
        ax0.plot(amount_data[column])
        ax0.set_title(f"Time Series for {column}")
        x_values_original = np.arange(1, len(amount_data[column]) + 1)
        ax0.set_xticks(x_values_original)

        # ACF plot
        ax1 = fig.add_subplot(gs[0, 1])
        plt.sca(ax1)
        plotting(acf_values)
        plt.title(f"ACF for {column}")

        # PACF plot
        ax2 = fig.add_subplot(gs[1, 1])
        plt.sca(ax2)
        plotting(pacf_values)
        plt.title(f"PACF for {column}")

        plt.tight_layout()
        plt.show()

# %% [markdown]
# Time lags where ACF or PACF exceeds confidence intervals will be the starting point for parameters selection MA and AR models, respectively.
# At time lag 0, ACF and PACF always indicate strong association between any given time lag, and itself. Given that, ACF and PACF at time lag 0 will not be considered.

# %% [markdown]
# ### Testing, fourier and SARIMA

# %%
from statsmodels.stats.diagnostic import acorr_ljungbox
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
from scipy.fft import fft, fftfreq

# %%
data = amount_data["VBR"]
data.plot()

# %%
data

# %%
data = {
    'time': np.arange(1, 24),
    'value': [
        -0.000005, -1.494895, 0.000028, -0.000029, -2.019398,
        0.000128, -0.000129, -2.151471, -0.000045, 0.000012,
        -2.913871, -0.000091, -0.000036, -2.433334, -0.000081,
        -0.000004, -2.272892, 0.000031, -0.000041, -2.205292,
        0.000083, -0.000014, -2.649706
    ]
}

df = pd.DataFrame(data)

plt.plot(df['time'], df['value'])
plt.title('Time Series Data')
plt.grid()
plt.show()


n = len(df['value'])
yf = fft(df['value'])
xf = fftfreq(n, 1)[:n//2]  # Only positive frequencies


plt.plot(xf, 2.0/n * np.abs(yf[:n//2])
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

# %%
# Decomposition

advanced_decomposition = STL(data['Value'], period = 3).fit()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True)

ax1.plot(advanced_decomposition.observed)
ax1.set_ylabel('Observed')

ax2.plot(advanced_decomposition.trend)
ax2.set_ylabel('Trend')

ax3.plot(advanced_decomposition.seasonal)
ax3.set_ylabel('Seasonal')

ax4.plot(advanced_decomposition.resid)
ax4.set_ylabel('Residuals')

fig.autofmt_xdate()
plt.tight_layout()

# %%
ARIMA_model = auto_arima(amount_data['VBR'], error_action='ignore', trace=True, suppress_warnings=True, seasonal=True, m = 4)
ARIMA_model.summary()

# %%
results.plot_diagnostics();
residuals = results.resid

# %%
acorr_ljungbox(residuals, np.arange(1, 2, 1))

# %%
SARIMA_model = SARIMAX(amount_data["VBR"], order=(1,1,1), seasonal_order=(2,0,1,2))
SARIMA_model_fit = SARIMA_model.fit(disp=False)
SARIMA_model_fit.plot_diagnostics(figsize=(10,8));

# %%
residuals = SARIMA_model_fit.resid

# %%
residuals

# %%
acorr_ljungbox(residuals, np.arange(1, 2, 1))

# %%



