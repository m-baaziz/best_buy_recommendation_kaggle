import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, cross_val_score

from metrics import custom_ap_at_k

def plot_learning_curves(estimator, title, x, y, cv):
	print('setting up learning curves')
	train_sizes=np.linspace(.1, 1.0, 5)

	plt.figure()
	plt.title(title)
	plt.xlabel('Training exemples')
	plt.ylabel('Score')

	print('computing scores')
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

def plot_cv_results(estimators, title, x, y, cv):
	print('getting cv results')
	cv_results = [cross_val_score(model[1], x, y, cv=cv) for model in estimators]
	print('plotting box plot')
	fig = plt.figure()
	plt.title(title)
	ax = fig.add_subplot(111)
	plt.boxplot(cv_results)
	ax.set_xticklabels([model[0] for model in estimators])
	return plt

def get_errors_input(X, Y, Y_pred, Y_pred_probas, classes):
	errors = [(X.loc[i, :].to_dict(), Y_pred[i], custom_ap_at_k(Y[i], Y_pred_probas[i], classes, 5), i) for i in range(len(X)) if Y[i] != Y_pred[i]]
	errors.sort(key = lambda x: x[2])
	return errors
