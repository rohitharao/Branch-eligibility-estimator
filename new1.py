import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import random as rn

# Set random seed for reproducibility
rn.seed(1254)
np.random.seed(42)
tf.random.set_seed(42)

# Load the CSV file
data = pd.read_csv('book3.csv')

# Preprocess the data
x = data.drop(columns=['Branch']).copy()
x['Rank'] = x['Rank'].astype(int)
le = LabelEncoder()
x['Gender'] = le.fit_transform(x['Gender'].apply(lambda x: 'Boy' if x == 'M' else 'Girl'))
x['Caste'] = le.fit_transform(x['Caste'].apply(lambda x: 'oc' if x == 'OC' else 'bc' if x == 'BC' else 'SC/ST'))
x['Rank'] = x['Rank'].astype(str).str.replace(',', '').astype(int)
y = le.fit_transform(data['Branch']).copy()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Define the ANN model using Keras
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(len(set(y)), activation='softmax'))

st.title('Colligate pathway predictor')

# Compile the Keras model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the Keras model
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)

# Get the user input
rank_input = st.text_input("Enter the student's rank: ")
if rank_input:
    try:
        rank = int(rank_input)
    except ValueError:
        st.write("Error: Please enter a valid integer for the rank.")
        rank = None
else:
    st.write("Note: Please enter a value for the rank.")
    rank = None

# Create radio buttons for selecting gender
gender = st.radio("Select the student's gender:", ('M', 'F'))

# Create radio buttons for selecting category
category = st.radio("Select the student's category:", ('OC', 'BC', 'SC', 'ST'))

# Preprocess the user input
user_input = np.array([[rank, 1 if gender == 'M' else 0, 1 if category == 'OC' else 0, 1.0]])

# Preprocess the user input using the same scaler used for training
user_input = scaler.transform(user_input)

# Create a submit button to trigger the prediction
if st.button('Submit'):
    # Use the trained Keras model to predict whether the entered student can get into the class based on their rank, gender, and category
    probabilities = model.predict(user_input)
    probabilities = probabilities / np.sum(probabilities)
    
    # Display the probability percentage of each branch
    branches=list(set(data['Branch']))
    branches.sort()
    sorted_probabilities=np.argsort(probabilities[0])[::-1]
    for i in sorted_probabilities:
        branch=branches[i]
        st.write(f"The probabilities of getting selected in {branch} is {probabilities[0][i]*100:.2f}%")
