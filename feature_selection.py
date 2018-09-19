import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion

import utils


class ItemSelector(BaseEstimator, TransformerMixin):
	def __init__(self, key, reshape=False):
		self.key = key
		self.reshape = reshape

	def fit(self, x, y=None):
		return self

	def transform(self, data, y=None):
		items = [item[self.key] for item in data]
		return np.array(items).reshape(-1, 1) if self.reshape else items

class LettersVectorizer(TfidfVectorizer):
	def build_tokenizer(self):
		tokenize = super(TfidfVectorizer, self).build_tokenizer()
		return lambda query: tokenize(utils.keep_letters(query))

class DigitsVectorizer(CountVectorizer):
	def build_tokenizer(self):
		tokenize = super(CountVectorizer, self).build_tokenizer()
		return lambda query: tokenize(utils.keep_digits(query))


def get_features_extractor():
	return FeatureUnion([
		('time_to_click', Pipeline([
			('time_to_click_selector', ItemSelector('time_to_click', True)),
			('normalizer', MaxAbsScaler())
		])),
		('bag_of_words', Pipeline([
			('query_selector', ItemSelector('query')),
			('vectorizers', FeatureUnion([
				('words', LettersVectorizer(analyzer='word', ngram_range=(1,2))),
				('chars', LettersVectorizer(analyzer='char_wb', ngram_range=(3,4))),
				('digits', DigitsVectorizer(analyzer='char', ngram_range=(1,2)))
			]))
		]))
	])















