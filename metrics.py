import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def custom_ap_at_k(actual, predicted_probas, classes, k):
	predicted_k_top_tuples = sorted(zip(predicted_probas, classes), reverse=True)[:k]
	top_labels = [item[1] for item in predicted_k_top_tuples]

	for i,item in enumerate(top_labels):
		if item == actual:
			return 1/(i + 1)

	return 0

def custom_map_at_k(y, y_pred_probs, classes):
	return np.mean([custom_ap_at_k(y[i], y_pred_probs[i], classes, 5) for i in range(0, len(y))])
