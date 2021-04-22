import pickle
import pandas as pd
 
#bring in serialized model
with open('model.pkl', 'rb') as f:
    classifier = pickle.load(f)
    
    
#function that uses model to predict the power generation 
def predict(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm):
    '''
    Input:
    sepal and petal dimensions - integer
 
    Output:
    flower species
    '''
    ds = pd.DataFrame([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]], 
                        columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm'])

    return classifier.predict(ds)[0]