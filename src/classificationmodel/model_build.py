
from flaml import AutoML
# import OS
import os
import pickle

def model_build(X_train,X_test,y_train,y_test):

    # Initialize an AutoML instance
    automl = AutoML()
    # Specify automl goal and constraint
    automl_settings = {
        "time_budget": 5,  # in seconds
        "metric": "accuracy",
        "task": "classification",
        "log_file_name": "models/model.log",
    }
    
    # Train with labeled input data
    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
    # Predict
    print(automl.predict_proba(X_train))
    # Print the best model
    print(automl.model.estimator) # type: ignore
    
    accuracy_test =  automl.score(X_test,y_test)
    accuracy_train = automl.score(X_train,y_train)
    
    # save model.pkl file  
    os.makedirs('models',exist_ok=True)
        
    with open('models/model.pkl','wb') as f:
        pickle.dump(automl,f)
            
    
    return accuracy_test,accuracy_train
