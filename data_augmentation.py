import string
import random
import time
import pandas as pd
from random import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from pydash import flatten
from pydash.strings import pad_start

from utils import remove_from_string
from data.utils import get_product_data, load_data, save_data

def random_time():
	return '2011-%s-%s %s:%s:%s.%s' % (
		pad_start(random.randint(1,12), 2, '0'),
		pad_start(random.randint(1,28), 2, '0'),
		pad_start(random.randint(0,23), 2, '0'),
		pad_start(random.randint(0,59), 2, '0'),
		pad_start(random.randint(0,59), 2, '0'),
		pad_start(random.randint(0,999), 3, '0'),
	)

def random_query_time(click_time, max_time_to_click_in_s):
	t = time.strptime(click_time, '%Y-%m-%d %H:%M:%S.%f')
	new_time = (time.mktime(t) + 2 * 60 * 60 + random.randint(0, max_time_to_click_in_s)) * 1000
	return time.strftime('%Y-%m-%d %H:%M:%S.',  time.gmtime(new_time/1000.)) + pad_start(random.randint(0,999), 3, '0')

def random_user():
	return ''.join(random.choice(string.ascii_lowercase + string.digits * 3) for _ in range(40))

def generate_data_from_product(sku, name):
	click_time = random_time()
	return [{
		'user': random_user(),
		'sku': sku,
		'category': 'abcat0701002',
		'query': remove_from_string(name, ' - Xbox 360'),
		'click_time': click_time,
		'query_time': random_query_time(click_time, 30)
	}]


def augment_data(train_filename, new_train_filename):
	products_data_map, products_name_map = get_product_data()
	data = load_data(train_filename)

	generated_data = flatten([generate_data_from_product(sku, product['name']) for sku, product in products_data_map.items()])
	generated_data_df = pd.DataFrame(generated_data, columns=['user', 'sku', 'category', 'query', 'click_time', 'query_time'])
	
	new_train = pd.concat([data, generated_data_df]).sample(frac=1).reset_index(drop=True)

	save_data(new_train, new_train_filename)

