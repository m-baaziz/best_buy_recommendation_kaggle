from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data.utils import get_product_data
from utils import timestring_to_ms


class DataTransformer(BaseEstimator, TransformerMixin):
	MAX_TIME_TO_CLICK = 60000

	def __init__(self, products_data_map, products_name_map):
		self.products_data_map = products_data_map
		self.products_name_map = products_name_map

	def fit(self, x, y=None):
		return self

	def get_time_to_click(self, item):
		return timestring_to_ms(item['click_time']) - timestring_to_ms(item['query_time'])

	def _is_valid_item(self, item):
		return not (item['sku'] in self.products_data_map and item['category'] not in self.products_data_map[item['sku']]['categories']) and \
			   self.get_time_to_click(item) <= self.MAX_TIME_TO_CLICK

	def _get_uniq_sku(self, sku):
		return self.products_name_map[self.products_data_map[sku]['name']]['sku'] if sku in self.products_data_map else sku # avoid doubles

	def transform(self, data, y=None):
		return [{
			'user': item['user'],
			'sku': self._get_uniq_sku(item['sku']),
			'query': item['query'],
			'time_to_click': self.get_time_to_click(item)
		} for item in data if self._is_valid_item(item)]

def preprocess_data(data):
	products_data_map, products_name_map = get_product_data()
	preprocess_pipeline = DataTransformer(products_data_map, products_name_map)
	x = preprocess_pipeline.fit_transform(data)
	y = [item['sku'] for item in x]
	return x, y
