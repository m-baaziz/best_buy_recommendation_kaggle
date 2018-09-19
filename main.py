import sys
import numpy as np
from random import shuffle
from pydash import flatten
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.metrics import classification_report
import feature_selection

from preprocessing import normalize, get_product_data, parse_data, load_data
from visualization import plot_learning_curves, get_errors_input
from metrics import custom_map_at_k


print 'Getting product data'
products_data_map, products_name_map = get_product_data()
print 'Loading data'
x_train, x_test, Y_train, Y_test = load_data(products_data_map, products_name_map, 'train_2.csv', 'test_2.csv')


print 'Seting up vectorizers'
vectorizer_1, vectorizer_2, vectorizer_3 = feature_selection.setup(x_train, products_data_map)

print 'Building X_train and X_test'
X_train_dense = normalize([feature_selection.build_feature_vector_dense(x) for x in x_train])
X_train_sparse = [feature_selection.build_feature_vector_sparse(x) for x in x_train]

X_test_dense = normalize([feature_selection.build_feature_vector_dense(x) for x in x_test])
X_test_sparse = [feature_selection.build_feature_vector_sparse(x) for x in x_test]

X_train = np.concatenate((X_train_dense, X_train_sparse), axis=1)
X_test = np.concatenate((X_test_dense, X_test_sparse), axis=1)


model_name = 'lr_1'
# model = joblib.load('./models/' + model_name + '_classifier.pkl')
print 'Fitting model'
model = LogisticRegression()
model.fit(X_train, Y_train)
print 'Saving model'
joblib.dump(model, './models/' + model_name + '_classifier.pkl')

print 'Predicting Y_pred'
Y_pred = model.predict(X_test)
Y_pred_probs = model.predict_proba(X_test)
errors_input = get_errors_input(x_test, Y_test, Y_pred)

print 'Scoring'
train_score = model.score(X_train, Y_train)
test_score = model.score(X_test, Y_test)

train_map_at_5 = custom_map_at_k(Y_train, model.predict_proba(X_train), model.classes_)
map_at_5 = custom_map_at_k(Y_test, Y_pred_probs, model.classes_)

print 'Reporting'
with open('./reports/' + model_name + '_report.txt', 'w') as report_file:
	report = classification_report(Y_test, Y_pred)
	report_file.write(report)
	report_file.write('\n\n Train Score : %s   Test Score : %s   MAP@5_Train : %s   MAP@5_Test : %s \n' % (train_score, test_score, train_map_at_5, map_at_5))
	report_file.write('\n\n\nErrors :\n\n')
	for index, item in enumerate(errors_input):
		report_file.write('%s  ---   %s   ---   %s\n' % (item[0], item[1], item[2]))
	report_file.write('\n\n\nBag of Words :\n\n')
	report_file.write('%s \n' % vectorizer_1.get_feature_names())
	report_file.write('%s \n' % vectorizer_2.get_feature_names())
	report_file.write('%s \n' % vectorizer_3.get_feature_names())

print 'Train Score : ', train_score
print 'Test Score : ', test_score
print 'MAP@5_Train : ', train_map_at_5
print 'MAP@5_Test : ', map_at_5

# cv = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)
# plot = plot_learning_curves(model, 'Learning Curves (Logistic Regression Classifier)', X_train, Y_train, cv)
# plot.show()
