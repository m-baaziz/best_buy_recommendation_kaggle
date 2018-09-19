import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import utils

vectorizer_1 = TfidfVectorizer(analyzer='word', ngram_range=(1,2))
vectorizer_2 = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,4))
vectorizer_3 = CountVectorizer(analyzer='char', ngram_range=(1,2))
products = None

def setup(data, products_data):
	global products
	products = products_data
	queries_letters = [utils.keep_letters(x['query']) for x in data]
	queries_digits = [utils.keep_digits(x['query']) for x in data]
	vectorizer_1.fit_transform(queries_letters)
	vectorizer_2.fit_transform(queries_letters)
	vectorizer_3.fit_transform(queries_digits)
	return vectorizer_1, vectorizer_2, vectorizer_3

def build_feature_vector_dense(x):
	return [ x['time_to_click'] ]

def transform_query(query):
	if not products:
		raise Exception('Feature selection not initialized')

	return np.array(
		list(vectorizer_1.transform([utils.keep_letters(query)]).toarray()[0]) +
		list(vectorizer_2.transform([utils.keep_letters(query)]).toarray()[0]) +
		list(vectorizer_3.transform([utils.keep_digits(query)]).toarray()[0])
	)

def build_feature_vector_sparse(x):
	return transform_query(x['query'])






















