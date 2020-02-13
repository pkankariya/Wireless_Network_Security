# Import libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import numpy
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from keras.utils import plot_model

seed = 9
numpy.random.seed(seed)

# Load input of datasets
#csv files were filtered based on the data.
input_file = "C:\\Users\\Poonam\\Desktop\\5G-dl\\data\\final.csv"
test_file = "C:\\Users\\Poonam\\Desktop\\5G-dl\\simdata\\testfile05072019.csv"

dataset = pd.read_csv(input_file).values

# Read training data
datasetTest = pd.read_csv(test_file).values

# Split the data set into input (X - feature predictors) and output (Y - target) variables
X = dataset[:,0:8].astype("int32")
Y = dataset[:,8]
XT = datasetTest[:,0:8].astype("int32")


encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# Convert integers to dummy variables using one hot encoding
dummy_y = np_utils.to_categorical(encoded_Y)

# Split the data sets into training and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(X, dummy_y, test_size=0.001, random_state=seed)
print(dummy_y)

# Creation of neural network model to prrocess the incoming request and identifying if it is genuine or a malicious threat
# Define model to be used
model = Sequential()
# Input layer
model.add(Dense(8, input_dim=8, init='normal', activation='relu'))
# Additional hidden layers
model.add(Dense(4, init='normal', activation='relu'))
model.add(Dense(3, init='normal', activation='tanh'))
# Output layer
model.add(Dense(3, init='normal', activation='softmax'))
# Identify the summary of the model defined and built
print('The summary of the neural network model built is ', model.summary())

# Compile the model defined
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model using the training data set
history = model.fit(X_train, Y_train, validation_split=0.1, epochs=16, batch_size=128)

# Evaluation of the performance of the model using the test data set
scores = model.evaluate(X_test, Y_test)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Graphical representation of the results
plot_model(model, to_file='model.png')

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

#from sklearn.metrics import confusion_matrix
#y_pred_keras = model.predict_classes(XT)

#csv = open("C:\\code\\fiveG\\simdata\\output.csv", "w") 
#"w" indicates that you're writing strings to the file

#pd.DataFrame(y_pred_keras).to_csv("C:\\code\\fiveG\\simdata\\output.csv")
#cm = confusion_matrix(Y_test, y_pred_keras, labels=[0, 1, 2])

#csv = open("C:\\code\\fiveG\\simdata\\input.csv", "w") 
#"w" indicates that you're writing strings to the file

#pd.DataFrame(XT).to_csv("C:\\code\\fiveG\\simdata\\input.csv")
