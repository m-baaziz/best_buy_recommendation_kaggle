import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

from metrics import custom_ap_at_k

def plot_learning_curves(estimator, title, x, y, cv):
	train_sizes=np.linspace(.1, 1.0, 5)

	plt.figure()
	plt.title(title)
	plt.xlabel('Training exemples')
	plt.ylabel('Score')

	train_sizes, train_scores, test_scores = learning_curve(estimator, x, y, cv=cv, train_sizes=train_sizes)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.grid()

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')
	plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
	plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')

	plt.legend(loc='best')
	return plt

def get_errors_input(X, Y, Y_pred, Y_pred_probas, classes):
	errors = [(v, Y_pred[i], custom_ap_at_k(Y[i], Y_pred_probas[i], classes, 5), i) for i,v in enumerate(X) if Y[i] != Y_pred[i]]
	errors.sort(key = lambda x: x[2])
	return errors
