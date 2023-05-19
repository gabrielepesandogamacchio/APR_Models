import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
   
DATASET_LOGMELSPEC_PATH = "data_logMelSpec_1sec_1600kh_64Mel.json"

def load_data(dataset_logMelSpec_path):
    with open(dataset_logMelSpec_path, "r") as fp:
        data = json.load(fp)

    #create numpy array
    inputs = np.array(data['logMelSpec'])
    targets = np.array(data['labels'])

    return inputs, targets


def plot_history(history):
    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    # save plot in current folder
    plt.savefig('training_history.png')


def plotConfusionMatrix(y_test, y_pred):
    # Load the data from the json file
    with open(DATASET_LOGMELSPEC_PATH, "r") as fp:
        data = json.load(fp)

    # Get the mapping
    mapping = data['mapping']

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize="true")

    plt.figure(figsize=(20, 20))

    # Plot the confusion matrix as a heatmap
    sns.heatmap(cm, annot=True, xticklabels=mapping, yticklabels=mapping)

    # Set the x-axis and y-axis labels
    plt.xlabel("Predicted class")
    plt.ylabel("True class")

    # Save the plot
    plt.savefig('confusion_matrix.png')


def prepare_datasets(test_size, validation_size):
    # load data
    X, y = load_data(DATASET_LOGMELSPEC_PATH)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size, stratify=y_train)

    # add an axis to input sets
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    # build network topology
    model = tf.keras.Sequential()

    # 1st conv layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))

    # output layer
    model.add(tf.keras.layers.Dense(8, activation='softmax'))

    return model

if __name__ == "__main__":
    # get train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)

    # compile model
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimiser,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=64, epochs=50)

    # plot accuracy/error for training and validation
    plot_history(history)

    # generate predictions for test set
    print("computing y_pred")
    y_pred = np.argmax(model.predict(X_test), axis=1)

    #plot confusion matrix
    plotConfusionMatrix(y_test, y_pred)

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
