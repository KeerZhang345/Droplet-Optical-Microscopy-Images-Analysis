import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List


def save_json(data: Dict, path: str) -> None:
    """
    Save to JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)

def load_json(path: str) -> Dict:
    """
    Load JSON file.
    """
    with open(path, "r") as f:
        return json.load(f)

def save_pickle(data: Dict, path: str) -> None:
    """
    Save to PICKLE.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_pickle(path: str) -> Dict:
    """
   Load PICKLE file.
   """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_csv(df: pd.DataFrame, path: str) -> None:
    """
    Save to csv.
    """
    df.to_csv(path, index=True)

def load_csv(path: str) -> pd.DataFrame:
    """
   Load csv file.
   """
    df = pd.read_csv(path)
    return df