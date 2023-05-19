import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
from sklearn.model_selection import KFold
import h5py

DATASET_LOGMELSPEC_PATH = "data_for_mobNET.h5"

def load_data(dataset_logMelSpec_path):
    with h5py.File(dataset_logMelSpec_path, "r") as fp:
        inputs = fp["logMelSpec"][:]
        targets = fp["labels"][:]

    return inputs, targets

def plot_history(history, save_path):
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

    # Save the plot as a PNG image
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def prepare_datasets(k_folds):
    # load data
    X, y = load_data(DATASET_LOGMELSPEC_PATH)

    # add an axis to input sets
    X = X[..., np.newaxis]

    # create K-fold splits
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    splits = kfold.split(X, y)

    # return generator that yields the train, validation, and test splits
    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # perform stratified sampling on the train set to create a validation set
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

        yield X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    model = tf.keras.Sequential()

    # block 1
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # block 2
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # block 3
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # block 4
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # flatten and dense layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(4096, activation='relu'))

    # output layer
    model.add(tf.keras.layers.Dense(8, activation='softmax'))

    return model


def plotConfusionMatrix(y_test, y_pred, save_path):
    # Get the mapping
    mapping = [
        "Calori",
        "Presenza Contemporanea di madri e capretti",
        "Isolamento sociale",
        "Separazione madre capretto",
        "Visita di estranei",
        "Distribuzione Cibo",
        "Ferita-Morte",
        "Fenomeni legati al parto"
    ]

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize="true")

    plt.figure(figsize=(20, 20))

    # Plot the confusion matrix as a heatmap
    sns.heatmap(cm, annot=True, xticklabels=mapping, yticklabels=mapping)

    # Set the x-axis and y-axis labels
    plt.xlabel("Predicted class")
    plt.ylabel("True class")

    # Save the plot as a PNG image
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    
if __name__ == "__main__":
    k_folds = 5
    # train model
    for i, (X_train, X_validation, X_test, y_train, y_validation, y_test) in enumerate(prepare_datasets(k_folds)):
        print(f"Training fold {i+1}...")

        # create new model instance for each fold
        model = build_model((128, 44, 1))

        # compile model
        optimiser = tf.keras.optimizers.Adam(learning_rate=0.00001)
        model.compile(optimizer=optimiser,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        
        model.summary()

        # train the model on this fold
        history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=12, epochs=14)

        # evaluate the model on the test set for this fold
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"Fold {i+1} test loss: {test_loss:.4f}")
        print(f"Fold {i+1} test accuracy: {test_acc:.4f}")

        # generate predictions on the test set for this fold
        y_pred = np.argmax(model.predict(X_test), axis=-1)

        # define the plots names
        imageNameMatrix = "Fold_" + str(i) + "_confusionMatrix.png"
        imageNameHistory = "Fold_" + str(i) + "_history.png"

        # plot confusion matrix for this fold
        plotConfusionMatrix(y_test, y_pred, imageNameMatrix)

        # plot the training history for the final fold
        plot_history(history, imageNameHistory)

