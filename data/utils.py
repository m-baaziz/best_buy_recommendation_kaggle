import os
import csv
import xmltodict
import sys

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


def parse_data(line):
	return {
		'user': line[0],
		'sku': line[1],
		'category': line[2],
		'query': line[3],
		'click_time': line[4],
		'query_time': line[5]
	}

def load_data(filename):
	with open(CSV_DIR + filename, 'r') as infile:
		reader = csv.reader(infile, delimiter=',')
		reader.__next__()
		return [parse_data(line) for line in reader]

def save_data(data, filename):
	with open(CSV_DIR + filename + '.csv', 'w') as fd:
		writer = csv.writer(fd, delimiter=',')
		writer.writerow(['user', 'sku', 'category', 'query', 'click_time', 'query_time'])
		for line in data:
			writer.writerow([
				line['user'],
				line['sku'],
				line['category'],
				line['query'],
				line['click_time'],
				line['query_time']
			])