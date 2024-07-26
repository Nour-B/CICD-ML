import pandas as pd 
from sklearn.preprocessing import StandardScaler

def load_data(filename: str) -> pd.DataFrame:
    """
    Reads the raw data file and returns pandas dataframe

    Parameters:
    filename (str): raw data filename

    Returns: 
    pd.DataFrame: pandas dataframe
    """
    df = pd.read_csv(filename)
    return df



def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs preprocessing steps on the data:
    1. Drop null values
    2. Convert categorical variables into dummy variables
    3. Scale the data to a normal distribution

    Parameters:

    Returns: 
    pd.DataFrame: preprocessed dataframe
    """

    # Drop null values
    #df = df.dropna()
    #df = df.drop([9,14])


    df = pd.get_dummies(df, dtype='int')
    #.drop('sex_.',axis=1)
    # dtype='int' ensures that the output will be 0/1 instead of True/False

    # Scale and fit with zero mean and unit variance
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    return pd.DataFrame(data=X,columns=df.columns)


def main():
    # Load the data
    penguins_df = load_data('raw_data/penguins.csv')

    preprocessed_penguins = preprocess_data(penguins_df)
    
    # Write processed dataset

    preprocessed_penguins.to_csv("processed_data/processed_penguins.csv", index=None)


if __name__ == "__main__":
    main()
 