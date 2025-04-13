# breast-cancer-prediction-mlp

This project demonstrates a simple deep learning approach for classifying breast cancer as malignant or benign using the Breast Cancer Wisconsin dataset. The model is built using TensorFlow and Keras, featuring a fully connected neural network with two hidden layers. The input data is standardized using StandardScaler, and the labels are one-hot encoded for categorical classification. The model is trained using categorical cross-entropy loss and the Adam optimizer. After 30 epochs of training, the model achieves a high level of accuracy on the test set. Training and validation losses are also visualized to assess the model's learning behavior over time.
# Importing TensorFlow library for building and training deep learning models
import tensorflow as tf

# Importing a built-in breast cancer dataset from scikit-learn
from sklearn.datasets import load_breast_cancer

# Importing function to split the dataset into training and test sets
from sklearn.model_selection import train_test_split

# Importing class to standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler

# Importing NumPy for numerical operations (not directly used here, but often needed)
import numpy as np

# Importing classes to build a Sequential model and dense (fully connected) layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Importing utility function to convert labels into one-hot encoded vectors
from tensorflow.keras.utils import to_categorical

# Importing categorical cross-entropy loss for multi-class classification problems
from tensorflow.keras.losses import CategoricalCrossentropy

# Importing matplotlib for data visualization and plotting graphs
import matplotlib.pyplot as plt

# Loading the breast cancer dataset
data = load_breast_cancer()

# Extracting the feature matrix (input data)
X = data.data

# Extracting the target labels (binary classification: malignant or benign)
Y = data.target

# Converting labels to one-hot encoded format (e.g., [1, 0] or [0, 1])
Y = to_categorical(Y, 2)

# Splitting data into training and test sets (80% training, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initializing a standard scaler to normalize the feature values
scaler = StandardScaler()

# Fitting the scaler on training data and transforming it
X_train = scaler.fit_transform(X_train)

# Using the same scaler to transform the test data
X_test = scaler.transform(X_test)

# Defining the architecture of the neural network using Sequential model
model = Sequential([
    # First hidden layer with 64 units and ReLU activation, input has 30 features
    Dense(units=64, activation='relu', input_shape=(30,)),
    # Second hidden layer with 32 units and ReLU activation
    Dense(units=32, activation='relu'),
    # Output layer with 2 units (for 2 classes) and softmax activation
    Dense(units=2, activation='softmax')
])

# Compiling the model with Adam optimizer and categorical cross-entropy loss
model.compile(
    optimizer='adam',
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']  # Tracking accuracy during training
)

# Training the model on the training data for 30 epochs with batch size of 32
history = model.fit(X_train, Y_train, epochs=30, batch_size=32, validation_data=(X_test, Y_test))

# Evaluating the trained model on the test dataset
loss, accuracy = model.evaluate(X_test, Y_test)

# Printing the final accuracy on test data as a percentage
print(f" Accuracy : {accuracy * 100:.2f}%")

# Plotting training and validation loss over epochs to visualize model performance
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Importing matplotlib to display images
import matplotlib.pyplot as plt

# Importing numpy to perform numerical operations (like argmax)
import numpy as np
class_name = ['Malignant', 'Benign']
index = 111

prediction = model.predict(X_test[index].reshape(1,-1))
predicted_class = class_name[np.argmax(prediction)]
print(f"Predicted: {predicted_class}")
print("Actual label:" , class_name[np.argmax(Y_test[index])])
