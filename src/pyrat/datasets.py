import gzip
import numpy as np
import os

import numpy as np

def train_val_split(X, y, val_size=0.1, random_state=None):
    """
    Splits a dataset into training and validation sets.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (num_samples, num_features).
    y : np.ndarray
        Label array of shape (num_samples, num_classes).
    val_size : float, optional (default=0.1)
        Proportion of the dataset to include in the validation split, between 0 and 1.
    random_state : int or None, optional (default=None)
        Seed for random shuffling to ensure reproducibility.

    Returns
    -------
    X_train : np.ndarray
        Training feature matrix.
    X_val : np.ndarray
        Validation feature matrix.
    y_train : np.ndarray
        Training labels.
    y_val : np.ndarray
        Validation labels.
    """
    # Ensure val_size is valid (i.e., a fraction between 0 and 1)
    if not (0 < val_size < 1):
        raise ValueError("test_size must be a float between 0 and 1")

    # Determine the total number of samples and the number allocated to validation
    num_samples = X.shape[0]
    val_size = int(val_size * num_samples)

    # Create an array of sample indices to shuffle
    indices = np.arange(num_samples)
    
    # If a random seed is provided, set it to ensure reproducible shuffling
    if random_state is not None:
        np.random.seed(random_state)
    np.random.shuffle(indices)  # In-place shuffle of indices

    # Split the indices into training set and validation set
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    # Use the split indices to partition X and y
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    return X_train, X_val, y_train, y_val


class MNIST:
    """
    A class to load and manage MNIST data (images and labels).
    
    Attributes
    ----------
    DATA_DIR : str
        The absolute path to the directory containing the MNIST `.gz` files.
    images : np.ndarray or None
        NumPy array of shape (num_samples, 28, 28, 1) containing normalized image data,
        or None if loading fails.
    labels : np.ndarray or None
        NumPy array of one-hot-encoded labels of shape (num_samples, 10), 
        or None if loading fails.
    is_training_set : bool
        Indicates whether this MNIST instance is for the training set or test set.
    num_samples : int
        Number of samples in this dataset (60,000 for train, 10,000 for test).
    """

    # Default directory for MNIST .gz files (train/test images/labels)
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/MNIST/"))

    def __init__(self, train=True):
        """
        Initialize the MNIST data loader.
        
        Parameters
        ----------
        train : bool, optional (default=True)
            If True, load the training set (60,000 samples); otherwise, load the test set (10,000 samples).
        """
        self.images = None
        self.labels = None
        self.is_training_set = train

        # Depending on whether this is the training or test set, set the number of samples
        # and load the corresponding images/labels.
        if train:
            self.num_samples = 60000
            self.images = self.__load_images('train-images-idx3-ubyte.gz')
            self.labels = self.__load_labels('train-labels-idx1-ubyte.gz')
        else:
            self.num_samples = 10000
            self.images = self.__load_images('t10k-images-idx3-ubyte.gz')
            self.labels = self.__load_labels('t10k-labels-idx1-ubyte.gz')

    def __len__(self):
        """
        Return the total number of samples in this dataset.
        """
        return self.num_samples

    def __str__(self):
        """
        String representation showing whether it's a training set, 
        how many samples it contains, and shape of images and labels.
        """
        return (f'Is training set: {self.is_training_set}\n'
                f'Number of samples: {self.num_samples}\n'
                f'Images shape: {self.images.shape}\n'
                f'Number of labels: {len(self.labels)}')

    def __load_images(self, file_name):
        """
        Load and preprocess image data from a gzip-compressed IDX file.

        Parameters
        ----------
        file_name : str
            Name of the gzipped IDX file containing image data.

        Returns
        -------
        data_norm : np.ndarray or None
            A NumPy array of shape (num_samples, 28, 28, 1) containing normalized image data,
            or None if file loading fails.
        """
        try:
            # Construct the full path to the file
            file_path = os.path.join(self.DATA_DIR, file_name)
            with gzip.open(file_path, 'r') as file:
                # The first 16 bytes in the IDX file contain metadata (magic number, sizes)
                file.read(16)
                # Read the raw bytes for all images: 28*28 pixels per image
                buf = file.read(28 * 28 * self.num_samples)
                # Convert bytes to a float32 NumPy array
                data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
                # Reshape into (num_samples, height, width, 1)
                data = data.reshape(self.num_samples, 28, 28, 1)
            # Normalize pixel values to range [0, 1]
            data_norm = data / 255.0
            return data_norm
        except FileNotFoundError:
            print(f"[ERROR] File not found, path: {file_path}")
        except Exception as e:
            print(f"[ERROR] Something happened during images loading: {e}")
        return None

    def __load_labels(self, file_name):
        """
        Load and one-hot-encode labels from a gzip-compressed IDX file.

        Parameters
        ----------
        file_name : str
            Name of the gzipped IDX file containing label data.

        Returns
        -------
        one_hot_labels : np.ndarray or None
            A NumPy array of shape (num_samples, 10) containing one-hot-encoded labels,
            or None if file loading fails.
        """
        try:
            # Construct the full path to the file
            file_path = os.path.join(self.DATA_DIR, file_name)
            with gzip.open(file_path, 'r') as file:
                # The first 8 bytes in the label IDX file contain metadata
                file.read(8)
                # Read the raw bytes for all labels
                buf = file.read(self.num_samples)
                # Convert bytes to a uint8 NumPy array
                labels = np.frombuffer(buf, dtype=np.uint8)
            # Create a one-hot matrix of shape (num_samples, 10)
            one_hot_labels = np.zeros((len(labels), 10))
            # Place '1' in the correct class index for each sample
            one_hot_labels[np.arange(len(labels)), labels] = 1
            return one_hot_labels
        except FileNotFoundError:
            print(f"[ERROR] File not found, path: {file_path}")
        except Exception as e:
            print(f"[ERROR] Something happened during labels loading: {e}")
        return None
