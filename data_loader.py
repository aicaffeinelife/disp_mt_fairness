from load_adult import adultDataset
from load_compas import load_compas
import pandas as pd
import numpy as np
from collections import namedtuple
from sklearn.preprocessing import StandardScaler

# TODO: DrugLoader, ArrythmLoader

def load_drug_data(dataset_file='datasets/drug_consumption.data'):
  '''
  Load drug consumption data with the target variable being
  consumption of heroin in the past.
  '''
  df = pd.read_csv(dataset_file)
  X = df.to_numpy()
  X_data = X[:, 1:12]
  y = X[:, 20]
  y[y == 'CL0'] = -1 # never used
  y[y != -1] = 1 # used at least once
  y = y.astype(np.double)
  sens = X[:, 5] # ethnicity
  sens[sens != -0.31685] = 0 # other races
  sens[sens == -0.31685] = 1 # white
  # print(np.unique(sens))

  scaler = StandardScaler()

  n_train = int(X_data.shape[0] * 0.75)
  X_train, y_train = X_data[:n_train, :], y[:n_train]
  sens_train = sens[:n_train]
  X_test, y_test = X_data[n_train:, :], y[n_train:]
  sens_test = sens[n_train:]
  X_train_sc = scaler.fit_transform(X_train)
  X_test_sc = scaler.fit_transform(X_test)
  data_train = namedtuple('_', ['data', 'target', 'sens'])(X_train_sc, y_train, sens_train)
  data_test = namedtuple('_', ['data', 'target', 'sens'])(X_test_sc, y_test, sens_test)

  return data_train, data_test


def load_arrhythmia_dataset(dataset_file='datasets/arrhythmia.data'):
  df = pd.read_csv(dataset_file)
  df.columns = [i for i in range(280)]
  df.drop(13, axis=1, inplace=True)
  df[[10, 11, 12, 14]] = df[[10, 11, 12, 14]].replace("?", np.NaN)
  df[[10, 11, 12, 14]] = df[[10, 11, 12, 14]].astype('float')
  df.fillna(df.median(), inplace=True)
  df.colums = [i for i in range(279)]
  data = df.values
  X = data[:, :278]
  y = data[:, 278]

  y[y != 1] = -1


  sens = data[:, 1]
  n_train = int(X.shape[0] * 0.75)
  X_train, y_train = X[:n_train, :], y[:n_train]
  sens_train = sens[:n_train]
  X_test, y_test = X[n_train:, :], y[n_train:]
  sens_test = sens[n_train:]

  scaler = StandardScaler()
  X_train_sc = scaler.fit_transform(X_train)
  X_test_sc = scaler.fit_transform(X_test)
  data_train = namedtuple('_', ['data', 'target', 'sens'])(X_train_sc,  y_train, sens_train)
  data_test = namedtuple('_', ['data', 'target', 'sens'])(X_test_sc, y_test, sens_test)

  return data_train, data_test


  # df = pd.read_csv(dataset_file)
  # X_arr = df.to_numpy()
  # X = X_arr[:, :-1]
  # y = X_arr[:, -1]
  # y[y != 1] = -1
  # sens = X[:, 1]
  # n_train = int(X.shape[0]*0.75)
  # X_train, y_train = X[:n_train, :], y[:n_train]
  # sens_train = sens[:n_train]
  # X_test, y_test = X[n_train:, :], y[n_train:]
  # sens_test = sens[n_train:]
  # data_train = namedtuple('_', ['data', 'target', 'sens'])(X_train,  y_train, sens_train)
  # data_test = namedtuple('_', ['data', 'target', 'sens'])(X_test, y_test, sens_test)
  # return data_train, data_test





def load_dataset(dataset_name):
  if dataset_name == 'adult':
    print('loading adult dataset...')
    data_train, data_test = adultDataset().load(mode='small', scaleby=0.5)
    sens_val = 'gender'
  elif dataset_name == 'compas':
    print('loading compas dataset...')
    data_train, data_test = load_compas()
    # data_train, data_test = CompasLoader()._process_data().load()
    sens_val = 'race'
  elif dataset_name == 'drug':
    print('loading drug dataset...')
    data_train, data_test = load_drug_data()
    sens_val = 'ethnicity'
  elif dataset_name == 'arrhythm':
    print('loading arrhythmia dataset...')
    data_train, data_test = load_arrhythmia_dataset()
    sens_val = 'gender'
  return data_train, data_test, sens_val

