# iris_code additions

I added a couple cells to the notebook - the last one is important because it serializes and exports the trained model
to the /mnt directory. This model is then used in the predictor.py script to create a model API. 

Model API - create an API using predictor.py as the script and predict as the function. An example input to use in the tester box is:
{
  "data": {
    "SepalLengthCm": 5.1,
    "SepalWidthCm": 3.5,
    "PetalLengthCm": 1.4,
    "PetalWidthCm": 0.2
  }
}

The iris_DecisionTree_hyperparameters.py script takes three command line arguments:

* max_depth 
* min_samples_leaf 
* min_samples_split 

Executed as a job (for example with `iris_RandomForest_estimators.py 3 1 2` or `iris_RandomForest_estimators.py 2 1 2`
it will output F1 score, Precision, and Recall. 