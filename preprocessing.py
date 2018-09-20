import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data.utils import get_product_data


class DataTransformer(BaseEstimator, TransformerMixin):
	MAX_TIME_TO_CLICK = 60000

	def __init__(self, products_data_map, products_name_map):
		self.products_data_map = products_data_map
		self.products_name_map = products_name_map

	def fit(self, x, y=None):
		return self

	def _is_valid_item(self, item):
		return not (item['sku'] in self.products_data_map and item['category'] not in self.products_data_map[item['sku']]['categories']) and \
			  	 item['time_to_click'] <= self.MAX_TIME_TO_CLICK

	def _get_uniq_sku(self, sku):
		return self.products_name_map[self.products_data_map[sku]['name']]['sku'] if sku in self.products_data_map else sku # avoid doubles

	def transform(self, df, y=None):
		new_df = df.copy()
		click_time_series = (pd.to_datetime(df['click_time'], unit='ms').astype(np.int64) // 10**6)
		query_time_series = (pd.to_datetime(df['query_time'], unit='ms').astype(np.int64) // 10**6)
		new_df.at[:, 'time_to_click'] = abs(click_time_series - query_time_series)
		return new_df.apply(lambda x: x if self._is_valid_item(x) else np.nan, axis=1).dropna().loc[:, ['user', 'sku', 'query', 'time_to_click']].reset_index(drop=True)


def preprocess_data(data):
	products_data_map, products_name_map = get_product_data()
	preprocess_pipeline = DataTransformer(products_data_map, products_name_map)
	x = preprocess_pipeline.fit_transform(data)
	y = x['sku']
	return x, y
