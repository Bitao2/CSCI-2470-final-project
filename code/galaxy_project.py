"""This is the main python file for the project.

The project can be run either by running this file directly, or by importing
this file in the project's jupyter notebook.
"""

import numpy as np
import convolution_model
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics

def split_data(images, labels, split_ratio):
    """Splits data into training, validation, and test sets.

    Args:
        images: Numpy array of shape (17736, 256, 256, 3).
        labels: Numpy array of shape (17736,).
        split_ratio: List of length 3, whose elements are each an integer
            in the range [1, 99], representing the percentages of the data to
            use for training, validation, and testing, respectively.

    Returns:
        A 6-tuple of numpy arrays representing the training images, training 
        labels, validation images, validationlabels, testing images, and 
        testing labels, respectively.
    """

    # Shuffle the data.
    rng = np.random.default_rng()
    permuted_indices = rng.permutation(len(labels))
    shuffled_img = images[permuted_indices]
    shuffled_lab = labels[permuted_indices]

    """
    # Split the data.  To deal with the fact that the given split_ratio values
    # might not correspond to integer-valued number of examples, we round
    # the training and validation sets to the nearest integer, and then just
    # use the remaining examples as the test set.
    num_ex = len(labels)
    num_train = round(num_ex * (split_ratio[0] * 0.01))
    num_valid = round(num_ex * (split_ratio[1] * 0.01))
    train_img = shuffled_img[:num_train,:,:]
    valid_img = shuffled_img[num_train:(num_train + num_valid),:,:]
    test_img = shuffled_img[(num_train + num_valid):,:,:]
    train_lab = shuffled_lab[:num_train]
    valid_lab = shuffled_lab[num_train:(num_train + num_valid)]
    test_lab = shuffled_lab[(num_train + num_valid):]
    """
    our_test_valid_size = (split_ratio[1] + split_ratio[2]) * 0.01
    split1 = StratifiedShuffleSplit(n_splits=1, test_size=our_test_valid_size, random_state=42)
    for train_index, test_valid_index in split1.split(images, labels):
        train_img, test_valid_img = images[train_index], images[test_valid_index]
        train_lab, test_valid_lab = labels[train_index], labels[test_valid_index]

    size_of_valid_as_frac_of_test_valid = (split_ratio[2] / (split_ratio[1] + split_ratio[2])) * 0.01
    split2 = StratifiedShuffleSplit(n_splits=1, test_size=size_of_valid_as_frac_of_test_valid, random_state=42)
    for test_index, valid_index in split2.split(test_valid_img, test_valid_lab):
        test_img, valid_img = test_valid_img[test_index], test_valid_img[valid_index]
        test_lab, valid_lab = test_valid_lab[test_index], test_valid_lab[valid_index]

    orig_class_ids, orig_class_counts = np.unique(labels, return_counts=True)
    train_class_ids, train_class_counts = np.unique(train_lab, return_counts=True)
    test_class_ids, test_class_counts = np.unique(test_lab, return_counts=True)
    valid_class_ids, valid_class_counts = np.unique(valid_lab, return_counts=True)

    num_orig_examples = images.shape[0]
    for orig_class_id, orig_class_count in enumerate(orig_class_counts):
        print(f"Class {orig_class_id} originally has {orig_class_count/num_orig_examples} fraction of examples.")
    num_valid_examples = valid_img.shape[0]
    for valid_class_id, valid_class_count in enumerate(valid_class_counts):
        print(f"Class {valid_class_id} originally has {valid_class_count/num_valid_examples} fraction of examples.")

    return train_img, train_lab, valid_img, valid_lab, test_img, test_lab



def run_cnn_model(images, labels, split_ratio=[70, 15, 15]):
    """Generates and trains a CNN model.

    Args:
        images: Numpy array of shape (17736, 256, 256, 3).
        labels: Numpy array of shape (17736,).
        split_ratio: List of length 3, whose elements are each an integer
            in the range [1, 99], representing the percentages of the data to
            use for training, validation, and testing, respectively. Defaults
            to [70, 15, 15].

    Returns:
        None
    """

    # Raise an error if the split_ratio values don't sum to 100.
    if split_ratio[0] + split_ratio[1] + split_ratio[2] != 100:
        raise AssertionError("Split ratio values don't sum to 100.")

    # Split the data into training, validation, and test sets.
    train_img, train_lab, valid_img, valid_lab, test_img, test_lab = split_data(
        images, labels, split_ratio)

    # Generate the CNN model.
    cnn_model = convolution_model.get_CNN_model()

    # Train the CNN model.
    print("Starting model training.")
    history = cnn_model.model.fit(
        train_img, train_lab,
        epochs = cnn_model.epochs,
        batch_size = cnn_model.batch_size,
        validation_data = (valid_img, valid_lab),
    )

    # Test the CNN model.
    print("Starting model testing.")
    test_results = cnn_model.model.evaluate(test_img, test_lab, cnn_model.batch_size)
    print("Test loss, test accuracy: ", test_results)

    # Print summary of the CNN model.
    cnn_model.model.summary()

    # Print confusion matrix.
    #predictions = cnn_model.model.predict(valid_img)
    #confusion_mat = cnn_model.model.get_confusion_matrix(test_lab, predictions)
    #print(confusion_mat)
    #fig_confusion1, ax_confusion1 = cnn_model.model.visualize_confusion_matrix(confusion_mat)
    #P1 = np.argmax(cnn_model.model.predict(valid_img), -1)
    #confusion_mtx = tf.math.confusion_matrix(P1, valid_lab)
    #P0 = np.argmax(cnn_model.model.predict(train_img), -1)
    #confusion_mtx = tf.math.confusion_matrix(P0, train_lab)

    """
    plt.figure(figsize=(12, 9))
    plt.imshow(confusion_mtx, cmap='hot', interpolation='nearest')

    fig, ax = plt.subplots(2, 10)
    fig.set_size_inches(24, 8)

    pred0 = cnn_model.predict(X0[:10])
    pred1 = cnn_model.predict(X1[:10])

    def p2l(pred):
        return D_info.features['label']._int2str[pred]

    for i in range(10):
        ax[0][i].imshow(X0[i], cmap = "Greys")
        ax[1][i].imshow(X1[i], cmap = "Greys")
        ax[1][i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax[0][i].set_xlabel(f"Pred {p2l(np.argmax(pred0[i], -1))} | {p2l(Y0[i])}")    
        ax[1][i].set_xlabel(f"Pred {p2l(np.argmax(pred1[i], -1))} | {p2l(Y1[i])}")

    """

    print(confusion_mtx)


    #confusion_matrix = metrics.confusion_matrix(valid_lab, predictions)
    #cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    #cm_display.plot()
    #plt.show()

    """
    fig, ax = plt.subplots(2, 10)
    fig.set_size_inches(24, 8)

    pred0 = cnn_model.predict(X0[:10])
    pred1 = cnn_model.predict(X1[:10])

    def p2l(pred):
        return D_info.features['label']._int2str[pred]

    for i in range(10):
        ax[0][i].imshow(X0[i], cmap = "Greys")
        ax[1][i].imshow(X1[i], cmap = "Greys")
        ax[1][i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax[0][i].set_xlabel(f"Pred {p2l(np.argmax(pred0[i], -1))} | {p2l(Y0[i])}")    
        ax[1][i].set_xlabel(f"Pred {p2l(np.argmax(pred1[i], -1))} | {p2l(Y1[i])}")
    """
    #plt.figure(figsize=(10, 8))
    #sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True, fmt='g')
    #plt.xlabel('Prediction')
    #plt.ylabel('Label')
    #plt.show()



