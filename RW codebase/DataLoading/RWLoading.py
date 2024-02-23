import pandas as pd
import sklearn.model_selection as sk
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def data_processing(data_file:str, columns, skip:int=0, date:str='date') -> pd.DataFrame:
    """
    A function used for data processing and visualization

    Arguments
    ----------
    data_file : str
        a formatted string to print out what the animal says
    columns : list[str]
        the name of the animal
    skip : int
        ignore the first skip rows
    date : str
        the sound that the animal makes
        
    Returned Values
    ----------
    data : pd.DataFrame

    """
    data = pd.read_csv(data_file)
    data[date] = pd.to_datetime(data[date])       #convert string to datetime
    data[date] = [d.date() for d in data[date]]   #convert datetime to date
    data = data.iloc[skip:]                       #keep index location skip to end
    data.reset_index(inplace = True, drop = True)  
    data = data.sort_values(by = date)            #sort by date just to make sure
    data = data[columns]                          #keep the column you want
    observed = data[['date','new_deaths']]
    return data, observed
def data_processing_ili(data_file:str, columns, skip:int=0, date:str='date') -> pd.DataFrame:
    """
    A function used for data processing and visualization

    Arguments
    ----------
    data_file : str
        a formatted string to print out what the animal says
    columns : list[str]
        the name of the animal
    skip : int
        ignore the first skip rows
    date : str
        the sound that the animal makes
        
    Returned Values
    ----------
    data : pd.DataFrame

    """
    data = pd.read_csv(data_file)
    data[date] = pd.to_datetime(data[date])       #convert string to datetime
    data[date] = [d.date() for d in data[date]]   #convert datetime to date
    data = data.iloc[skip:]                       #keep index location skip to end
    data.reset_index(inplace = True, drop = True)  
    data = data.sort_values(by = date)            #sort by date just to make sure
    data = data[columns]                          #keep the column you want
    observed = data[['date','ILITOTAL']]
    return data, observed

def plot_data(data_file:str):
    return data.plot(subplots = True, figsize = (10, 12))


"""
data_splitting: 
    A module for splitting data.

Functions
----------
train_test_split: 
    def train_test_split(x: np.ndarray, y: np.ndarray, test_ratio: float = 0.25) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_test_split(x: np.ndarray, y: np.ndarray, test_ratio: float = 0.25) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    
    """ 
    A function used for splitting the data into fixed train and test sets.
    Usually, for time series, it is better to use a rolling window.
    For the train/test split, we use the default split ratio by scikit-learn 
    set to 0.25.

    Arguments
    ----------
    x : pd.DataFrame
        past lagged data
        
    y : pd.DataFrame
        future horizons
        
    test_ratio : int
        the fraction of data for testing purposes
        
    Returned Values
    ----------
    x_train : numpy.ndarray
    x_test : numpy.ndarray
    y_train : numpy.ndarray
    y_test : numpy.ndarray
    
    """ 
    x_train, x_test, y_train, y_test = sk.train_test_split(x, y, test_size = test_ratio, 
                                                           random_state = 42, shuffle = False)
    return x_train, x_test, y_train, y_test


def make_input_output_sequences(series, n_past, n_future, include_dates):
    X, y = list(), list()
    forecast, target = list(), list()
    for window_start in range(n_past,len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end-n_past > len(series):
            break
        past = series[(window_start-n_past):(past_end-n_past), :]
        future = series[(past_end-n_past):(future_end-n_past), :]
        X.append(np.array([past]).T.tolist())
        y.append(np.array([future]).T.tolist())
        x = np.array(X)
    x = np.squeeze(np.swapaxes(np.array(X), 1, 2), axis=-1)
    y = np.squeeze(np.swapaxes(np.array(y), 1, 2), axis=-1)
    if include_dates is True:
        return np.array(x[:,:,:]), np.array(y[:,:,:])
    else:
        return torch.from_numpy(np.array(x[:,:,1:],dtype=np.float32)).float().to(device), torch.from_numpy(np.array(y[:,:,1:],dtype=np.float32)).float().to(device)

def shift_sequence(x_train, y_train, x_test, y_test, window, include_dates):
    x_train = x_train[window:,:,:]
    x_train_list = x_train.tolist()
    for i in range (0,window):
        x_train_list.append(x_test[i,:,:].tolist())
    x_train = np.array(x_train_list)
    y_train = y_train[window:,:,:]
    y_train_list = y_train.tolist()
    for i in range(0,window):
        y_train_list.append(y_test[i,:,:].tolist())
    y_train = np.array(y_train_list)
    x_test = x_test[window:,:,:]
    y_test = y_test[window:,:,:]

    if include_dates is True:
        return x_train, x_test, y_train, y_test
    else:
        return torch.from_numpy(x_train).to(device).float(), x_test.to(device).float(), torch.from_numpy(y_train).to(device).float(), y_test.to(device).float()





"""
data_transforms: 
    A module for transforming data

Functions
----------
data_transform_std: 
    def data_transform_std(df: pd.DataFrame, test_ratio: float = 0.7):
    
data_transform_minmax
    def data_transform_minmax(df: pd.DataFrame, test_ratio: float = 0.7, min_: float = 0, max_: float = 1):
"""
def data_transform_std(df: pd.DataFrame, test_ratio: float = 0.7):
    
    """
    A function used for data transformation to make sure it's 
    in the sensitive active region of the activation function
    by substracting the mean and dividing by the standard deviation

    Arguments
    ----------
    data : pd.DataFrame
        the input data
    test_ratio : int
        the fraction of data for testing purposes
        
    Returned Values
    ----------
    scalers : dict

    """ 
    scalers={}
    for i in range(0, len(df.columns)):
        if (i == 0):
            continue
        scaler = StandardScaler()
        scaler.fit(df.iloc[0:round(test_ratio*df.shape[0]), i].values.reshape(-1, 1))
        rescaled = scaler.transform(df.iloc[:, i].values.reshape(-1, 1))
        scalers['scaler_' + df.columns[i]] = scaler
        df.iloc[:, i] = pd.DataFrame(rescaled)
    return scalers, df 

def data_transform_minmax(df: pd.DataFrame, test_ratio: float = 0.7, min_: float = 0, max_: float = 1):
    
    """
    A function used for data transformation to make sure it's 
    in the sensitive active region of the activation function
    by rescaling the data to be between min_ and max_

    Arguments
    ----------
    data : pd.DataFrame
        the input data
    test_ratio : int
        the fraction of data for testing purposes
        
    Returned Values
    ----------
    scalers : dict

    """ 
    scalers={}
    for i in range(0, len(df.columns)):
        if (i == 0):
            continue
        scaler = MinMaxScaler(feature_range=(min_, max_))
        scaler.fit(df.iloc[0:round(test_ratio*df.shape[0]), i].values.reshape(-1, 1))
        rescaled = scaler.transform(df.iloc[:, i].values.reshape(-1, 1))
        scalers['scaler_' + df.columns[i]] = scaler
        df.iloc[:, i] = pd.DataFrame(rescaled)
    return scalers, df 