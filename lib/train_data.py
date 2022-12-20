import pandas as pd
from joblib import dump, load
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


class TrainData:
    
    def __init__(self, params):

        self.input_x_train = params.get("input_x_train")
        self.input_y_train = params.get("input_y_train")

        self.output_model = params.get("output_model")


    
    def execute(self):
        
        df_x = pd.read_csv(self.input_x_train)
        df_y = pd.read_csv(self.input_y_train)

        
        x = df_x.values
        y = df_y.values
        
        clf = MLPClassifier(random_state=1, max_iter=3000)
        clf.fit(x,y)
        dump(clf, self.output_model)
        

        
        
    
        