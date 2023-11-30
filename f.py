import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import Tk, Label, Button
from tkinter import filedialog

def prompt_user(question):
    return int(input(question))

def start_process():
    return input('Type y to start the process, or type n to terminate the process: ')

def calculate_cost(items, cost_per_item):
    return items * cost_per_item

def perform_regression(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build a simple neural network for regression using Keras
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    # Evaluate the model on the test set
    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    # Visualize the predictions vs. actual values
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=predictions.flatten())
    plt.title('Actual vs. Predicted Luminosity')
    plt.xlabel('Actual Luminosity')
    plt.ylabel('Predicted Luminosity')
    plt.show()

def create_gui():
    root = Tk()
    root.title("Clothing Wash Analysis")

    def load_data():
        file_path = filedialog.askopenfilename()
        if file_path:
            data = pd.read_csv(file_path)
            print(data.head())

            # Perform regression on the loaded data
            X = data[['Shirts', 'Pants', 'Undergarment', 'Dresses', 'Other']]
            y = data['Total Cost']
            perform_regression(X, y)

    label = Label(root, text="Clothing Wash Analysis")
    label.pack()

    load_button = Button(root, text="Load Data", command=load_data)
    load_button.pack()

    root.mainloop()

def main():
    print('Welcome to Casual Clothing Wash')

    laundry_count = prompt_user('Enter laundry count: ')

    shirts = prompt_user('How many shirts/t-shirts: ')
    pants = prompt_user('How many shorts/pants/trousers: ')
    undergarment = prompt_user('How many undergarments - underwear, bra, tank-top, socks: ')
    dresses = prompt_user('How many dresses/skirts: ')
    other = prompt_user('Other articles of clothing: ')

    if laundry_count == (shirts + pants + undergarment + dresses + other):
        begin_process = start_process()

        if begin_process == 'y':
            print('Washing your clothes!')

            shirts_cost = calculate_cost(shirts, 0.50)
            pants_cost = calculate_cost(pants, 0.65)
            undergarment_cost = calculate_cost(undergarment, 0.25)
            dresses_cost = calculate_cost(dresses, 0.75)
            other_cost = calculate_cost(other, 0.45)

            total_cost = shirts_cost + pants_cost + undergarment_cost + dresses_cost + other_cost

            print(f'We are charging you $ {total_cost} for {laundry_count} pieces of clothing')

            confirm = input('Type granted in the terminal to confirm the transfer, or type terminate to end the program: ')

            if confirm == 'granted':
                print('Transfer successful, have a nice day!')
            else:
                print('We will care for your clothes while you get the certain amount of money, have a nice day!')
                sys.exit()
        else:
            print('Terminating program, have a nice day!')
            sys.exit()

    else:
        print('Hmm... Looks like the total laundry count does not add up to the articles of clothing you gave... Please run the program again!')
        sys.exit()

if __name__ == "__main__":
    main()
    create_gui()
