import os
import csv
import xmltodict
import sys
import pandas as pd
import numpy as np

import sys
sys.path.append(os.path.abspath('..'))
from best_buy_recommendation_kaggle.utils import timestring_to_ms

CSV_DIR = os.path.abspath('./data/csv/') + '/'
XML_DIR = os.path.abspath('./data/xml/') + '/'

def get_product_data():
	with open(XML_DIR + 'small_product_data.xml', 'r') as infile:
		raw_doc = xmltodict.parse(infile.read())
		products_id_data_map = { product['sku']: {
			'name': product['name'],
			'price': float(product['salePrice']),
			'release_date': timestring_to_ms(product['startDate']),
			'preowned': int(product['preowned'] == 'true'),
			'customer_review_count': int(product['customerReviewCount'] or 0),
			'customer_review_average': float(product['customerReviewAverage'] or 0),
			'categories': [category['id'] for category in product['categoryPath']['category']]
		} for product in raw_doc['products']['product'] }

		products_name_id_map = { product['name']: {
			'sku': product['sku']
		} for product in raw_doc['products']['product'] }

		return products_id_data_map, products_name_id_map

def load_data(filename):
	return pd.read_csv(CSV_DIR + filename, dtype={'sku': str}, parse_dates=[4, 5])

def save_data(df, filename):
	df.to_csv(CSV_DIR + filename, index=False)

