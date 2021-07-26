#import packages
import json
import pandas as pd
import numpy as np
import sys
 
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
 
max_depth = int(sys.argv[1])
min_samples_leaf = int(sys.argv[2])
min_samples_split = int(sys.argv[3])

#read in data.
input_file = "iris_orig.csv"
iris_df = pd.read_csv(input_file, header = 0)
 
#format data 
feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
X = iris_df.loc[:, feature_cols]
y = iris_df.Species
 
#split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
#set a random seed so results will be the same for all of us
np.random.seed(415)
 
#fit model on training data 
my_model = tree.DecisionTreeClassifier(max_depth=max_depth, 
                                       min_samples_leaf=min_samples_leaf,
                                       min_samples_split=min_samples_split)
my_model.fit(X_train,y_train)
 
#determine predictions
predictions = my_model.predict(X_test)
 
#output metrics - here we chose precision and recall
precision = metrics.precision_score(y_true = y_test, y_pred = predictions, average ='weighted')
recall = metrics.recall_score(y_true = y_test, y_pred = predictions, average ='weighted')
f1_score = metrics.f1_score(y_true = y_test, y_pred = predictions, average ='weighted')
 
#output metrics to dominostats for display in jobs tab
with open('dominostats.json', 'w') as f:
    f.write(json.dumps({"Precision": precision, "Recall": recall, "F1_score": f1_score}))
