import numpy as np
import pickle

# Load the model saved
loaded_model = pickle.load(open("C:/Users/sunka/trained_model.sav",'rb'))

# Making a predictive system
input_data=(5,37,382,128,25.8,0.23,54,97)

#converting the input data into numpy array
input_data_array=np.asarray(input_data)

#reshape the array as we are predicting 
input_data_reshaped=input_data_array.reshape(1,-1)

prediction=loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
    print("PERSON NON DIABETIC")
else:
    print("PERSON IS DIABETIC")