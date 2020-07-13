import os
import numpy as np
import keras

# Stack RGB channels of 5 frames and generate (batch_size,128,128,15) input array
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, data_dir, batch_size=16, dim=(128,128), n_channels=3, n_stack=5, shuffle=True): # paper was batch_size=64
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs # list_IDs = (0, 1, ..., -n_stack) exclude edges of list for frame stacking
        self.data_dir = data_dir
        self.n_channels = n_channels
        self.n_stack = n_stack
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        input_data_dir = os.path.join(self.data_dir, 'input_data')
        gt_data_dir = os.path.join(self.data_dir, 'gt_data')

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels*self.n_stack))
        y = np.empty((self.batch_size, *self.dim, self.n_channels))
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X_stack = np.empty((*self.dim, self.n_channels*self.n_stack))
            
            # TODO: better to make list from idx 0 to len-(n_stack)
            for j in range(-(self.n_stack//2), self.n_stack//2+1):

                X_stack[:,:,self.n_channels*(j+self.n_stack//2):self.n_channels*(j+self.n_stack//2+1)] = np.load(os.path.join(input_data_dir, str(int(ID) + j).zfill(8) + '.npy'))
                
            # Store class
            X[i,] = X_stack
            y[i,] = np.load(os.path.join(gt_data_dir, ID + '.npy'))

        X /= 255
        y /= 255

        return X, y


def load_batch_data(list_IDs_temp, data_dir='data', batch_size=16, dim=(128,128), n_channels=3, n_stack=5):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

    input_data_dir = os.path.join(data_dir, 'input_data')
    gt_data_dir = os.path.join(data_dir, 'gt_data')

    # Initialization
    X = np.empty((batch_size, *dim, n_channels*n_stack))
    y = np.empty((batch_size, *dim, n_channels))

    # Generate data
    for i, ID in enumerate(list_IDs_temp):
        # Store sample
        X_stack = np.empty((*dim, n_channels*n_stack))

        for j in range(-(n_stack//2), n_stack//2+1):

            X_stack[:,:,n_channels*(j+n_stack//2):n_channels*(j+n_stack//2+1)] = np.load(os.path.join(input_data_dir, str(int(ID) + j).zfill(8) + '.npy'))

        # Store class
        X[i,] = X_stack
        y[i,] = np.load(os.path.join(gt_data_dir, ID + '.npy'))

    X /= 255
    y /= 255

    return X, y