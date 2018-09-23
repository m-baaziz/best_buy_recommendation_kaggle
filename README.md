# Products Recommendation (Best buy data set)

Best Buy Xbox games recommendation using scikit-learn and pandas


## Overview

The goal of this project is to build a model that can predict which products (xbox games) a user would be interested in, based on the search query.

data source : https://www.kaggle.com/c/acm-sf-chapter-hackathon-small/data , including :

- train.csv : training data set
- small_product_data.xml : information about the xbox products

## Raw data schema

Items of the raw data provided by the kaggle challenge describe a user's click on a single product.
Each line is composed of :

- user: A user ID
- sku: The stock-keeping unit (item) that the user clicked on
- category: The category the sku belongs to
- query: The search terms that the user entered
- click_time: Time the sku was clicked on
- query_time: Time the query was run

## Data Augmentation

Some games are not so popular and some never appear on the training set. To make sure that the model will be trained to recognize every games, the training set has been augmented with entries for each xbox game (cf. data_augmentation.py)

## Preprocessing

To be relevant, the training data set had to go through some preprocessing steps :

- Transform click_time and query_time into number of milliseconds from epoch.
- Add a new column : time_to_click = | click_time - query_time |.
- The same game may appear twice with two different IDs (sku). Replace duplicate ID (sku) with the original.
- Check if the product category described in the data set entry matches the one in the product description (small_product_data.xml). If not, remove the entry.
- Check if the product has been clicked on less than 1 minute after the query was made. If not, remove the entry. This allows us to reduce the number of false positives (where the user clicks on a product after navigating on the website, and not because of the query he or she made).

## Features Selection

Here are the features selected to train the model :

- time_to_click
- bag of words : 
	- by words (ngram range : 1 to 2)
	- by characters (ngram range 3 to 4)
	- by digits (ngram range 1 to 2)

## Metrics

The metrics used to discriminate between different models were :

- classification accuracy
- precision/recall and f1-score
- a custom made MAP@5 : order the predicted probalities for a test entry, from the most likely to the least. Take the first 5 predicted targets. How far is the true target in this subset of 5 ? If it is the first : the score is 1 for this entry. the second -> 0.5 , the third -> 0.33 ... If it is not among the 5 -> 0. Finaly, we compute the mean of this score for each entry. This metric ranges from 0 to 1 and behaves similarily to the classification accuracy.

## Selected Model

After comparing different classification algorithms, the Logistic Regression (C = 1) proved to give the best results.

**Model comparaison :**

![Project sketch](https://github.com/m-baaziz/best_buy_recommendation_kaggle/blob/master/reports/model_comparaison.png)

**Learning Curves :**

![Project sketch](https://github.com/m-baaziz/best_buy_recommendation_kaggle/blob/master/reports/learning_curves.png)

## Results

Train Score : 0.7888348554936149   Test Score : 0.7669995244888255   MAP@5 on Training set : 0.8749621107289237   MAP@5 on Testing set : 0.854979658688646 
