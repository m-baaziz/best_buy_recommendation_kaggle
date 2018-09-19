import sys
import numpy as np
from random import shuffle
from pydash import flatten
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from data.utils import load_data
from preprocessing import preprocess_data
from visualization import plot_learning_curves, get_errors_input
from metrics import custom_map_at_k
from feature_selection import get_features_extractor


print 'Loading data'
train_data = load_data('train_2.csv')
test_data = load_data('test_2.csv')

print 'Preprocessing'
X_train, Y_train = preprocess_data(train_data)
X_test, Y_test = preprocess_data(test_data)

model_name = 'lr_'

# print 'Loading model'
# model = joblib.load('./models/' + model_name + '_classifier.pkl')
print 'Fitting model'
model = Pipeline([
	('features', get_features_extractor()),
	('LogisticRegression', LogisticRegression())
])
model.fit(X_train, Y_train)
print 'Saving model'
joblib.dump(model, './models/' + model_name + '_classifier.pkl')

print 'Predicting Y_pred'
Y_pred = model.predict(X_test)
Y_pred_probs = model.predict_proba(X_test)
errors_input = get_errors_input(X_test, Y_test, Y_pred, Y_pred_probs, model.classes_)

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
	report_file.write('\n\n\nErrors :\n')
	report_file.write('\nPattern :  input --- predicted label --- AP@5 --- input index\n\n')
	for index, item in enumerate(errors_input):
		report_file.write('%s  ---   %s   ---   %s   ---   %s\n' % (item[0], item[1], item[2], item[3]))

print 'Train Score : ', train_score
print 'Test Score : ', test_score
print 'MAP@5_Train : ', train_map_at_5
print 'MAP@5_Test : ', map_at_5

# cv = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)
# plot = plot_learning_curves(model, 'Learning Curves (Logistic Regression Classifier)', X_train, Y_train, cv)
# plot.show()
