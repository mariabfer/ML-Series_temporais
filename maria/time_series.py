from __future__ import annotations
import pandas as pd
import os
import glob

class TimeSeries:
    def __init__(self, df: pd.DataFrame, series_name: str):
        self.original = df
        self.splits = { 'original': df }
        self.name = series_name
        self.data = {}

    def update_splits(self, splitter: TimeSeriesSplitter, ) -> None:
        splits = splitter.split(self.original)

        if 'original' in splits.keys():
            raise Exception("Split key cannot be reserved key 'original'.")

        self.splits = {
            'original': self.original,
            **splits
        }
        
    def __getitem__(self, key: str) -> pd.DataFrame:
        if key not in self.splits.keys():
            raise Exception("Invalid split key.")
    
        return self.splits[key]


class TimeSeriesIO:
    @staticmethod
    def load_csv(file_path: str, series_name: str, read_csv_kwargs: dict = {}):
        if not os.path.exists(file_path):
            raise Exception(f"Cannot find a file at specified path (\"{file_path}\").")
        
        df = pd.read_csv(file_path, **read_csv_kwargs)
        return TimeSeries(df, series_name)

    @staticmethod
    def load_folder_csv(folder_path: str, read_csv_kwargs: dict = {}):
        if not os.path.exists(folder_path):
            raise Exception(f"Cannot find a folder at specified path (\"{folder_path}\").")

        file_paths = glob.glob(f"{folder_path}/*.csv")
        print(f"Founded:\n{'\n\t-'.join(file_paths)}")
        series = {}
        for file_path in file_paths:
            series_name = os.path.basename(file_path)[:-4]
            series[series_name] = TimeSeriesIO.load_csv(file_path, series_name, read_csv_kwargs)
        
        return series

class TimeSeriesSplitter:
    def __init__(self, splits: dict[str, float]):
        if any(v <= 0 for v in splits.values()):
            raise ValueError("Splits proportions must be positive.")
        if abs(sum(splits.values()) - 1) > 1e-10:
            raise ValueError("Splits proportions must sum to 1.")
        
        self.splits = splits
    
    def split(self, df: pd.DataFrame):
        splitted = {}
        n = len(df)

        last_end = 0
        keys = list(self.splits.keys())

        for i, key in enumerate(keys):
            if i == len(keys) - 1:
                # Use all remaining data.
                splitted[key] = df.iloc[last_end:]
            else:
                # Split using proportion.
                proportion = self.splits[key]
                size = int(n * proportion)
                splitted[key] = df.iloc[last_end:last_end + size]
                last_end += size

        return splitted


class RandomWalkUtilities:
    @staticmethod
    def extract_random_walk(ts: TimeSeries, split: str, target: str):
        target_series = ts[split][target]
        rw_series = target_series.shift(1)
        rw_residuals = target_series - rw_series

        df_dict = {
            "rw_residuals": rw_residuals,
            "rw": rw_series
        }
        df = pd.DataFrame(df_dict, index=target_series.index)
        return TimeSeries(df, f"RW_{ts.name}")

