import csv
import xmltodict
from utils import timestring_to_ms
import sklearn.preprocessing
from sklearn.model_selection import train_test_split

DATA_DIR = './data/'
MAX_TIME_TO_CLICK = 60000

def normalize(x):
	return sklearn.preprocessing.MaxAbsScaler().fit_transform(x)

def get_product_data():
	with open(DATA_DIR + '/small_product_data.xml', 'r') as infile:
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

def get_raw_data(filename='data.csv'):
	with open(DATA_DIR + filename, 'r') as infile:
		reader = csv.reader(infile, delimiter=',')
		reader.next()

		data = [{
			'user': item[0],
			'sku': item[1],
			'category': item[2],
			'query': item[3],
			'click_time': item[4],
			'query_time': item[5]
		} for item in reader]

		Y = [x['sku'] for x in data]
		return data, Y

def is_valid_training_data(item, products_data_map, filename):
	return not (item[1] in products_data_map and item[2] not in products_data_map[item[1]]['categories']) and \
		   timestring_to_ms(item[4]) - timestring_to_ms(item[5]) <= MAX_TIME_TO_CLICK


def parse_data(products_data_map, products_name_map, filename='train.csv'):
	with open(DATA_DIR + filename, 'r') as infile:
		reader = csv.reader(infile, delimiter=',')
		reader.next()

		data = []
		Y = []

		for item in reader:
			if is_valid_training_data(item, products_data_map, filename):
				sku = products_name_map[products_data_map[item[1]]['name']]['sku'] if item[1] in products_data_map else item[1] # avoid doubles
				data.append({
					'user': item[0],
					'sku': sku,
					'query': item[3],
					'query_time': timestring_to_ms(item[5]),
					'time_to_click': timestring_to_ms(item[4]) - timestring_to_ms(item[5])
				})
				Y.append(sku)

		return data, Y

def save_data(data, filename):
	with open(DATA_DIR + filename + '.csv', 'w') as fd:
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

def load_data(products_data_map, products_name_map, train_filename='train.csv', test_filename='test.csv'):
	x_train, y_train = parse_data(products_data_map, products_name_map, train_filename)
	x_test, y_test = parse_data(products_data_map, products_name_map, test_filename)
	return x_train, x_test, y_train, y_test


