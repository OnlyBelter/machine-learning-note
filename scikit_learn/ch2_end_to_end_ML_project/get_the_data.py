import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = 'datasets/housing'
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + '/housing.tgz'


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    download raw data and unpack
    :param housing_url:
    :param housing_path:
    :return:
    """
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


def split_train_test(data, test_radio):
    """
    split raw into two parts with totally random method, training data and test data
    :param data:
    :param test_radio:
    :return:
    """
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_radio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    """
    split raw data by using hash
    :param data:
    :param test_ratio:
    :param id_column:
    :param hash:
    :return:
    """
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


# fetch_housing_data()
housing = load_housing_data()
print(housing.head())
print(housing.info())
print(housing['ocean_proximity'].value_counts())
print(housing.describe())
# housing.hist(bins=50, figsize=(20, 15))
# plt.show()

#  -------sampling-------
#  -- randomly split raw data
# train_set, test_set = split_train_test(housing, 0.2)
# print(len(train_set), 'train +', len(test_set), 'test')
#  -- hash
# housing_with_id = housing.reset_index()
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'index')

#  -- using sk-learn
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# print(split.split(housing, housing['income_cat']))  # a generator
# strat_train_set = pd.DataFrame()
# strat_test_set = pd.DataFrame()
for train_index, test_index in split.split(housing, housing['income_cat']):
    print(len(train_index), len(test_index))
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(housing['income_cat'].value_counts() / len(housing))
for set in (strat_train_set, strat_test_set):
    set.drop(['income_cat'], axis=1, inplace=True)

housing = strat_train_set.copy()

#  - a geographical scatter plot of the data (only training data)
#  - figure 2-13, California housing prices
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
             s=housing['population']/100, label='population',
             c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
# plt.legend()
# plt.show()
corr_matrix = housing.corr()
print(corr_matrix)
print(corr_matrix['median_house_value'].sort_values(ascending=False))

#  - figure 2-15. Scatter matrix
attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()

#  - figure 2-16. Median income versus median house value
# housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.2)
# plt.show()

encoder_label = LabelEncoder()
housing_cat = housing['ocean_proximity']
housing_cat_encoded = encoder_label.fit_transform(housing_cat)  # 先转换成序号表示
encoder_hot = OneHotEncoder()
# the output is a SciPy sparse matrix
housing_cat_1hot = encoder_hot.fit_transform(housing_cat_encoded.reshape(-1, 1))
print(type(housing_cat_1hot))  # Compressed Sparse Row format
print(housing_cat_1hot.toarray())  # one-hot 的形式, a dense NumPy array














