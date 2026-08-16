from src.classificationmodel.model_build import model_build
from src.classificationmodel.data_ingestion import load_data
from src.classificationmodel.data_preprocessing import preprocessing




def main():
    
    # Step1 : Load the Data
    df = load_data()
    print(df.shape)
    
    # Step2 : Data Pre processing
    X_train,X_test,y_train,y_test,transformer = preprocessing(df)
    
    #
    accuracy_test,accuracy_train= model_build(X_train,X_test,y_train,y_test)
    
    print(f'Training Accuracy : {accuracy_train}')
    print(f'Testing Accuracy : {accuracy_test}')
    


main()