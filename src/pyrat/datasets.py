import gzip
import numpy as np
import os

class MNIST:
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/MNIST/"))

    def __init__(self, train=True):
        self.images = None
        self.labels = None
        self.is_training_set = train

        if train:
            self.num_samples = 60000
            self.images = self.__load_images('train-images-idx3-ubyte.gz')
            self.labels = self.__load_labels('train-labels-idx1-ubyte.gz')
        else:
            self.num_samples = 10000
            self.images = self.__load_images('t10k-images-idx3-ubyte.gz')
            self.labels = self.__load_labels('t10k-labels-idx1-ubyte.gz')

    def __len__(self):
        return self.num_samples

    def __str__(self):
        return f'Is training set: {self.is_training_set}\nNumber of samples: {self.num_samples}\nImages shape: {self.images.shape}\nNumber of labels: {len(self.labels)}'

    # Metodo privato per caricare le immagini
    def __load_images(self, file_name):
        try:
            file_path = os.path.join(self.DATA_DIR, file_name)  # Percorso completo al file
            with gzip.open(file_path, 'r') as file:
                file.read(16)  # Salta l'header
                buf = file.read(28 * 28 * self.num_samples)  # Carica le immagini
                data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
                data = data.reshape(self.num_samples, 28, 28, 1)  # Reshape a (num_samples, 28, 28, 1)
            data_norm = data / 255.0
            return data_norm
        except FileNotFoundError:
            print(f"[ERROR] File not found, path: {file_path}")
        except Exception as e:
            print(f"[ERROR] Something happened during images loading: {e}")
        return None

    # Metodo privato per caricare le etichette
    def __load_labels(self, file_name):
        try:
            file_path = os.path.join(self.DATA_DIR, file_name)  # Percorso completo al file
            with gzip.open(file_path, 'r') as file:
                file.read(8)  # Salta l'header
                buf = file.read(self.num_samples)  # Carica le etichette
                labels = np.frombuffer(buf, dtype=np.uint8)
            # One-hot encoding delle etichette
            one_hot_labels = np.zeros((len(labels), 10))  # 10 classi per MNIST
            one_hot_labels[np.arange(len(labels)), labels] = 1
            return one_hot_labels
        except FileNotFoundError:
            print(f"[ERROR] File not found, path: {file_path}")
        except Exception as e:
            print(f"[ERROR] Something happened during labels loading: {e}")
        return None
