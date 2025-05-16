# data_loader.py
import pandas as pd

class DataLoader:
    """
    Loads a dataset from a CSV (or TSV) file into a pandas DataFrame.

    Attributes:
        path (str): Filesystem path or URL to the dataset.
        delimiter (str): Character delimiting fields (e.g., ',', '\t').
        encoding (str): File encoding (e.g., 'utf-8', 'latin-1').
    """
    def __init__(self, path: str, delimiter: str = ',', encoding: str = 'utf-8'):
        """
        Initializes the DataLoader.

        Args:
            path (str): Path or URL to load the dataset from.
            delimiter (str): Separator for pandas.read_csv; any single character.
            encoding (str): Text encoding for file read.
        """
        self.path = path
        self.delimiter = delimiter
        self.encoding = encoding

    def load(self) -> pd.DataFrame:
        """
        Reads the CSV/TSV file into a pandas DataFrame.

        Returns:
            pd.DataFrame: Loaded dataset.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If parsing errors occur.
        """
        return pd.read_csv(self.path, delimiter=self.delimiter, encoding=self.encoding)
