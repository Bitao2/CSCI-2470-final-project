"""This is the main python file for the project.

The project can be run either by running this file directly, or by importing
this file in the project's jupyter notebook.
"""

import numpy as np
import convolutional_model

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
    cnn_model = convolutional_model.get_CNN_model()

    # Train the CNN model.
    print("Starting model training.")
    history = cnn_model.model.fit(
        train_img, train_lab,
        epochs = epochs,
        batch_size = batch_size,
        validation_data = (valid_img, valid_lab),
    )

    # Test the CNN model.
    print("Starting model testing.")
    test_results = cnn_model.model.evaluate(test_img, test_lab, batch_size)
    print("Test loss, test accuracy: ", test_results)

    # Print summary of the CNN model.
    cnn_model.model.summary()


