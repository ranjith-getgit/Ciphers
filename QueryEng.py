# ============================ Packages ============================
import os
import sys
import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import duckdb
import redis
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import StringIO

# ============================ 1. Data Ingestion Layer ============================
# Handles loading of financial data from CSV or API.

class DIE:

    def __init__(self, cache):
        """Initialize with a shared Redis cache instance."""
        self.cache = cache

    def load_csv(self, file_path):
        """Loads data from a CSV file using Dask."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")
        try:
            self.cache.flushdb()
            data = dd.read_csv(file_path)
            print(f"CSV file '{file_path}' loaded successfully.")
            return data
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")

    def load_api(self, ticker, period="1y", interval="1d"):
        """Loads stock data from Yahoo Finance."""
        if not ticker:
            raise ValueError("Ticker symbol must be provided.")
        try:
            df = yf.download(ticker, period=period, interval=interval)
            if df.empty:
                raise ValueError(f"No data found for ticker: {ticker}")
            self.cache.flushdb()
            print(f"Stock data for {ticker} loaded successfully.")
            return dd.from_pandas(df, npartitions=4)  # Convert to Dask DataFrame
        except Exception as e:
            raise ValueError(f"Error loading stock data: {e}")

    def stock_info(self, ticker):
        """Fetches metadata about a stock ticker."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if "regularMarketPrice" not in info:
                raise ValueError(f"Failed to fetch stock info for {ticker}.")
            return {
                "Company": info.get("longName", "N/A"),
                "Current Price": info.get("regularMarketPrice", "N/A"),
                "52-Week High": info.get("fiftyTwoWeekHigh", "N/A"),
                "Market Cap": info.get("marketCap", "N/A"),
            }
        except Exception as e:
            raise ValueError(f"Error fetching stock info: {e}")

# ============================ 2. Data Storage & Transformation Layer ============================
# Handles storage and transformation of data for querying.

class DSTO:

    def __init__(self, dfqe):
        """Converts and registers data for querying with DuckDB."""
        self.dfqe_pd = dfqe.compute()  # Convert Dask to Pandas
        self.dfqe_pd.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in self.dfqe_pd.columns]
        self.dfqe_pa = pa.Table.from_pandas(self.dfqe_pd)  # Convert to PyArrow Table for better interoperability
        duckdb.register("dfqe", self.dfqe_pa)

    def pan_dataframe(self):
        """Returns the Pandas DataFrame."""
        return self.dfqe_pd

# ============================ 3. Query Execution Layer ============================
# Handles SQL queries on the financial dataset using DuckDB and Redis caching.

class QEN:

    def __init__(self, dfqe):
        self.cache = redis.Redis(host='localhost', port=6379, db=0)
        self.data = dfqe  # DuckDB table already registered

    def query(self, sql_qe):
        """Executes an SQL query with caching."""
        cached = self.cache.get(sql_qe)
        if cached:
            print('Returning cached result.')
            return pd.read_json(StringIO(cached.decode("utf-8")))
        print('Executing query...')
        result = duckdb.query(sql_qe).df()

        self.cache.set(sql_qe, result.to_json()) # Cache the result
        return result

# ============================ 4. Presentation Layer ============================
# Handles user input, querying, and visualization.

class UI:

    def __init__(self, query_engine):
        self.query_engine = query_engine

    def query_loop(self):
        """Interactive loop for user SQL queries and plotting."""
        while True:
            print("\nEnter your SQL query (or type 'back' to reload data, 'exit' to quit):")
            user_query = sys.stdin.readline().strip()  # Accepts long queries properly
            if user_query.lower() == 'exit':
                print("Exiting program..")
                sys.exit()
            if user_query.lower() == 'back':
                print("Returning to data ingestion..")
                return "back"
            try:
                output = self.query_engine.query(user_query)
                print("\nQuery Output:\n", output)
                self.plot_data(output)
            except Exception as e:
                print(f"Error executing query: {e}. Please try again.")

    def plot_data(self, output):
        """Handles user input for plotting stock data."""
        while True:
            user_input = input("\nEnter column names to plot (comma-separated, excluding 'Date'), 'back' to return, or 'exit' to quit: ").strip()
            if user_input.lower() == "exit":
                print("Exiting program..")
                sys.exit()
            if user_input.lower() == "back":
                print("Returning to SQL query input..")
                break
            col1 = ["Date"] + [col.strip() for col in user_input.split(",") if col.strip()]
            col2 = [col for col in col1 if col != "Date"]
            invalid_cols = [col for col in col2 if col not in output.columns]
            if invalid_cols:
                print(f"Error: Invalid columns: {', '.join(invalid_cols)}. Please try again.")
                continue
            self.plot_stock_data(output, col2)

    def plot_stock_data(self, output, columns):
        """Generates dynamic subplots for selected stock columns."""
        n_plots = len(columns)
        n_cols = min(2, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7, 6 * n_rows))
        axes = [axes] if n_plots == 1 else axes.flatten()
        for i, col in enumerate(columns):
            sns.lineplot(x=output["Date"], y=output[col], ax=axes[i], color="black")
            axes[i].set_title(f"{col} Stock Prices")
            axes[i].set_xlabel("Date")
            axes[i].set_ylabel(col)
            axes[i].grid(True, linestyle="--")
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.show()

# ============================ Main Execution ============================

if __name__ == "__main__":
    query_engine = QEN(None)  # Initialize QueryEngine to create cache

    while True:  # Loop to allow re-ingestion
        ingestion = DIE(query_engine.cache)
        dfqe = None  # Initialize dfqe to avoid NameError

        # Load data from API or CSV
        while dfqe is None:  # Keep asking until valid data is loaded
            data_source = input("Enter 'csv' to load from CSV, 'api' for Yahoo Finance, or 'exit' to quit: ").strip().lower()
            if data_source == "exit":
                print("Exiting program..")
                sys.exit()  # Ensure the script exits
            if data_source == "csv":
                while True:
                    file_path = input("Enter the full CSV file path or type 'back' to return: ").strip()
                    if file_path.lower() == "back":
                        break  # Go back to the data source selection
                    try:
                        dfqe = ingestion.load_csv(file_path)
                        break  # Proceed if CSV loads successfully
                    except Exception as e:
                        print(f"Error loading CSV: {e}. Try again.")
            elif data_source == "api":
                while True:
                    ticker = input("Enter stock ticker (e.g., AAPL) or type 'back' to return: ").strip()
                    if ticker.lower() == "back":
                        break  # Go back to the data source selection
                    try:
                        dfqe = ingestion.load_api(ticker=ticker, period="1y", interval="1d")
                        stock_info = ingestion.stock_info(ticker)
                        print(stock_info)
                        break  # Proceed if API data loads successfully
                    except Exception as e:
                        print(f"Error loading stock data: {e}. Try again.")
            else:
                print("Invalid input. Please enter 'csv', 'api', or 'exit'.")
                continue  # Restart the loop if input is invalid

        # Convert data for querying
        storage = DSTO(dfqe)

        # Re-initialize QueryEngine with new data
        query_engine = QEN(storage.pan_dataframe())

        # Start the interactive SQL and plotting UI
        ui = UI(query_engine)
        user_choice = ui.query_loop()

        # If user wants to go back to ingestion, restart the loop
        if user_choice == "back":
            continue  # Restart from the ingestion level
        else:
            break  # Exit loop when done
