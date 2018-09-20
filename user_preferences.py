from collections import defaultdict
from utils import dict_list_mean_by_key, dict_list_mode_by_key

def build_user_preferences(history, products):
	preferences = {}

	mean_product = {
		'price': dict_list_mean_by_key(products, 'price'),
		'release_date': dict_list_mean_by_key(products, 'release_date'),
		'preowned': int(dict_list_mean_by_key(products, 'preowned') >= 0.5),
		'customer_review_count': dict_list_mean_by_key(products, 'customer_review_count'),
		'customer_review_average': dict_list_mean_by_key(products, 'customer_review_average')
	}

	for line in history:
		if not line['sku'] in products:
			products[line['sku']] = mean_product
		product = products[line['sku']]
		if not line['user'] in preferences:
			preferences[line['user']] = {
				'price': product['price'],
				'time_from_release': line['query_time'] - product['release_date'],
				'preowned': product['preowned'],
				'customer_review_count': product['customer_review_count'],
				'customer_review_average': product['customer_review_average'],
				'click_count': 1
			}
		else:
			preferences[line['user']]['price'] += product['price']
			preferences[line['user']]['time_from_release'] += line['query_time'] - product['release_date']
			preferences[line['user']]['preowned'] += product['preowned']
			preferences[line['user']]['customer_review_count'] += product['customer_review_count']
			preferences[line['user']]['customer_review_average'] += product['customer_review_average']
			preferences[line['user']]['click_count'] += 1

	for user_id, user_pref in preferences.items():
		user_pref['price'] /= user_pref['click_count']
		user_pref['time_from_release'] /= user_pref['click_count']
		user_pref['preowned'] = int(user_pref['preowned'] >= user_pref['click_count'] / 2)
		user_pref['customer_review_count'] /= user_pref['click_count']
		user_pref['customer_review_average'] /= user_pref['click_count']

	mean_user = {
		'price': dict_list_mode_by_key(preferences, 'price'),
		'time_from_release': dict_list_mode_by_key(preferences, 'time_from_release'),
		'preowned': dict_list_mode_by_key(preferences, 'preowned'),
		'customer_review_count': dict_list_mode_by_key(preferences, 'customer_review_count'),
		'customer_review_average': dict_list_mode_by_key(preferences, 'customer_review_average'),
		'click_count': dict_list_mode_by_key(preferences, 'click_count')
	}

	return defaultdict(lambda: mean_user, preferences)
