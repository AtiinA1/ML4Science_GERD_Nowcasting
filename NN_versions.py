#!/usr/bin/env python
# coding: utf-8

# In[1]:

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import LearningRateScheduler


def scheduler(epoch, lr):
    """
    Learning rate scheduler function for training neural networks.

    This scheduler function adjusts the learning rate based on the training epoch. It keeps the learning rate constant for
    the first 20 epochs and then applies an exponential decay with a decay rate of tf.math.exp(-0.1).

    Parameters:
        epoch (int): The current training epoch.
        lr (float): The current learning rate.

    Returns:
        float: The adjusted learning rate for the current epoch.
    """
    if epoch < 20:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
    
    
def NLP_width_64(X_train, y_train, input_shape,initial_rate=0.001,batch_size = 8,epochs=80):
    
    """
    Create and train a more complex neural network model for Natural Language Processing (NLP) tasks.

    This function defines a multi-layer neural network model with several hidden layers, ReLU activation functions,
    and uses mean squared error as the loss function. It also includes a learning rate scheduler callback to adjust
    the learning rate during training.

    Parameters:
        X_train (numpy.ndarray): Input training data.
        y_train (numpy.ndarray): Target training data.
        input_shape (int): The shape of the input data.

    Returns:
        tensorflow.keras.models.Sequential: The trained neural network model.
    """
    #Fix randomness 
    tf.random.set_seed(675)
    
    # More complex neural network model
    model = Sequential([
        Dense(64, input_shape=(input_shape,)),
        Activation('relu'),
        Dense(1)
    ])

    # Compile the model with an initial learning rate
    initial_learning_rate = initial_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(scheduler)

    # Fit the model with the learning rate scheduler
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler], validation_split=0.2)
    
    # Calculate the Mean Absolute Error on the test set
    return model

def NLP_width_128(X_train, y_train, input_shape,initial_rate=0.001,batch_size = 8,epochs=80):
    """
    Create and train a more complex neural network model for Natural Language Processing (NLP) tasks.

    This function defines a multi-layer neural network model with several hidden layers, ReLU activation functions,
    and uses mean squared error as the loss function. It also includes a learning rate scheduler callback to adjust
    the learning rate during training.

    Parameters:
        X_train (numpy.ndarray): Input training data.
        y_train (numpy.ndarray): Target training data.
        input_shape (int): The shape of the input data.

    Returns:
        tensorflow.keras.models.Sequential: The trained neural network model.
    """
    #Fix randomness 
    tf.random.set_seed(675)
    
    
    # More complex neural network model
    model = Sequential([
        Dense(128, input_shape=(input_shape,)),
        Activation('relu'),
        Dense(1)
    ])

    # Compile the model with an initial learning rate
    initial_learning_rate = initial_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(scheduler)

    # Fit the model with the learning rate scheduler
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler], validation_split=0.2)
    
    # Calculate the Mean Absolute Error on the test set
    return model

def NLP_width_256(X_train, y_train, input_shape,initial_rate=0.001,batch_size = 8,epochs=80):
    """
    Create and train a more complex neural network model for Natural Language Processing (NLP) tasks.

    This function defines a multi-layer neural network model with several hidden layers, ReLU activation functions,
    and uses mean squared error as the loss function. It also includes a learning rate scheduler callback to adjust
    the learning rate during training.

    Parameters:
        X_train (numpy.ndarray): Input training data.
        y_train (numpy.ndarray): Target training data.
        input_shape (int): The shape of the input data.

    Returns:
        tensorflow.keras.models.Sequential: The trained neural network model.
    """
    
    #Fix randomness 
    tf.random.set_seed(675)
    
    
    # More complex neural network model
    model = Sequential([
        Dense(256, input_shape=(input_shape,)),
        Activation('relu'),
        Dense(1)
    ])

    # Compile the model with an initial learning rate
    initial_learning_rate = initial_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(scheduler)

    # Fit the model with the learning rate scheduler
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler], validation_split=0.2)
    
    # Calculate the Mean Absolute Error on the test set
    return model

def NLP_width_512(X_train, y_train, input_shape,initial_rate=0.001,batch_size = 8,epochs=80):
    """
    Create and train a more complex neural network model for Natural Language Processing (NLP) tasks.

    This function defines a multi-layer neural network model with several hidden layers, ReLU activation functions,
    and uses mean squared error as the loss function. It also includes a learning rate scheduler callback to adjust
    the learning rate during training.

    Parameters:
        X_train (numpy.ndarray): Input training data.
        y_train (numpy.ndarray): Target training data.
        input_shape (int): The shape of the input data.

    Returns:
        tensorflow.keras.models.Sequential: The trained neural network model.
    """
    
    #Fix randomness 
    tf.random.set_seed(675)
    
    # More complex neural network model
    model = Sequential([
        Dense(64, input_shape=(input_shape,)),
        Activation('relu'),
        Dense(1)
    ])

    # Compile the model with an initial learning rate
    initial_learning_rate = initial_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(scheduler)

    # Fit the model with the learning rate scheduler
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler], validation_split=0.2)
    
    # Calculate the Mean Absolute Error on the test set
    return model

def NLP_width_1024(X_train, y_train, input_shape,initial_rate=0.001,batch_size = 8,epochs=80):
    """
    Create and train a more complex neural network model for Natural Language Processing (NLP) tasks.

    This function defines a multi-layer neural network model with several hidden layers, ReLU activation functions,
    and uses mean squared error as the loss function. It also includes a learning rate scheduler callback to adjust
    the learning rate during training.

    Parameters:
        X_train (numpy.ndarray): Input training data.
        y_train (numpy.ndarray): Target training data.
        input_shape (int): The shape of the input data.

    Returns:
        tensorflow.keras.models.Sequential: The trained neural network model.
    """
    
    
    #Fix randomness 
    tf.random.set_seed(675)
    
    # More complex neural network model
    model = Sequential([
        Dense(1024, input_shape=(input_shape,)),
        Activation('relu'),
        Dense(1)
    ])

    # Compile the model with an initial learning rate
    initial_learning_rate = initial_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(scheduler)

    # Fit the model with the learning rate scheduler
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler], validation_split=0.2)
    
    # Calculate the Mean Absolute Error on the test set
    return model

def NLP_width_2048(X_train, y_train, input_shape,initial_rate=0.001,batch_size = 8,epochs=80):
    """
    Create and train a more complex neural network model for Natural Language Processing (NLP) tasks.

    This function defines a multi-layer neural network model with several hidden layers, ReLU activation functions,
    and uses mean squared error as the loss function. It also includes a learning rate scheduler callback to adjust
    the learning rate during training.

    Parameters:
        X_train (numpy.ndarray): Input training data.
        y_train (numpy.ndarray): Target training data.
        input_shape (int): The shape of the input data.

    Returns:
        tensorflow.keras.models.Sequential: The trained neural network model.
    """
    
    
    
    #Fix randomness 
    tf.random.set_seed(675)
    
    # More complex neural network model
    model = Sequential([
        Dense(2048, input_shape=(input_shape,)),
        Activation('relu'),
        Dense(1)
    ])

    # Compile the model with an initial learning rate
    initial_learning_rate = initial_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(scheduler)

    # Fit the model with the learning rate scheduler
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler], validation_split=0.2)
    
    # Calculate the Mean Absolute Error on the test set
    return model

def NLP_width_4096(X_train, y_train, input_shape,initial_rate=0.001,batch_size = 8,epochs=80):
    """
    Create and train a more complex neural network model for Natural Language Processing (NLP) tasks.

    This function defines a multi-layer neural network model with several hidden layers, ReLU activation functions,
    and uses mean squared error as the loss function. It also includes a learning rate scheduler callback to adjust
    the learning rate during training.

    Parameters:
        X_train (numpy.ndarray): Input training data.
        y_train (numpy.ndarray): Target training data.
        input_shape (int): The shape of the input data.

    Returns:
        tensorflow.keras.models.Sequential: The trained neural network model.
    """
    
    #Fix randomness 
    tf.random.set_seed(675)
    
    
    # More complex neural network model
    model = Sequential([
        Dense(4096, input_shape=(input_shape,)),
        Activation('relu'),
        Dense(1)
    ])

    # Compile the model with an initial learning rate
    initial_learning_rate = initial_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(scheduler)

    # Fit the model with the learning rate scheduler
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler], validation_split=0.2)
    
    # Calculate the Mean Absolute Error on the test set
    return model

def NLP_width_8192(X_train, y_train, input_shape,initial_rate=0.001,batch_size = 8,epochs=80):
    """
    Create and train a more complex neural network model for Natural Language Processing (NLP) tasks.

    This function defines a multi-layer neural network model with several hidden layers, ReLU activation functions,
    and uses mean squared error as the loss function. It also includes a learning rate scheduler callback to adjust
    the learning rate during training.

    Parameters:
        X_train (numpy.ndarray): Input training data.
        y_train (numpy.ndarray): Target training data.
        input_shape (int): The shape of the input data.

    Returns:
        tensorflow.keras.models.Sequential: The trained neural network model.
    """
    
    #Fix randomness 
    tf.random.set_seed(675)
    
    
    
    # More complex neural network model
    model = Sequential([
        Dense(8192, input_shape=(input_shape,)),
        Activation('relu'),
        Dense(1)
    ])

    # Compile the model with an initial learning rate
    initial_learning_rate = initial_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(scheduler)

    # Fit the model with the learning rate scheduler
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler], validation_split=0.2)
    
    # Calculate the Mean Absolute Error on the test set
    return model

def NLP_width_16384(X_train, y_train, input_shape,initial_rate=0.001,batch_size = 8,epochs=80):
    """
    Create and train a more complex neural network model for Natural Language Processing (NLP) tasks.

    This function defines a multi-layer neural network model with several hidden layers, ReLU activation functions,
    and uses mean squared error as the loss function. It also includes a learning rate scheduler callback to adjust
    the learning rate during training.

    Parameters:
        X_train (numpy.ndarray): Input training data.
        y_train (numpy.ndarray): Target training data.
        input_shape (int): The shape of the input data.

    Returns:
        tensorflow.keras.models.Sequential: The trained neural network model.
    """
    
    #Fix randomness 
    tf.random.set_seed(675)
       
    # More complex neural network model
    model = Sequential([
        Dense(16384, input_shape=(input_shape,)),
        Activation('relu'),
        Dense(1)
    ])

    # Compile the model with an initial learning rate
    initial_learning_rate = initial_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(scheduler)

    # Fit the model with the learning rate scheduler
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler], validation_split=0.2)
    
    # Calculate the Mean Absolute Error on the test set
    return model

def NLP_width_32768(X_train, y_train, input_shape,initial_rate=0.001,batch_size = 8,epochs=80):
    """
    Create and train a more complex neural network model for Natural Language Processing (NLP) tasks.

    This function defines a multi-layer neural network model with several hidden layers, ReLU activation functions,
    and uses mean squared error as the loss function. It also includes a learning rate scheduler callback to adjust
    the learning rate during training.

    Parameters:
        X_train (numpy.ndarray): Input training data.
        y_train (numpy.ndarray): Target training data.
        input_shape (int): The shape of the input data.

    Returns:
        tensorflow.keras.models.Sequential: The trained neural network model.
    """
    
    #Fix randomness 
    tf.random.set_seed(675)
    
    # More complex neural network model
    model = Sequential([
        Dense(32768, input_shape=(input_shape,)),
        Activation('relu'),
        Dense(1)
    ])

    # Compile the model with an initial learning rate
    initial_learning_rate = initial_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(scheduler)

    # Fit the model with the learning rate scheduler
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler], validation_split=0.2)
    
    # Calculate the Mean Absolute Error on the test set
    return model


def NLP_depth_11(X_train, y_train, input_shape,initial_rate=0.001,batch_size = 8,epochs=80):
    """
    Create and train a more complex neural network model for Natural Language Processing (NLP) tasks.

    This function defines a multi-layer neural network model with several hidden layers, ReLU activation functions,
    and uses mean squared error as the loss function. It also includes a learning rate scheduler callback to adjust
    the learning rate during training.

    Parameters:
        X_train (numpy.ndarray): Input training data.
        y_train (numpy.ndarray): Target training data.
        input_shape (int): The shape of the input data.

    Returns:
        tensorflow.keras.models.Sequential: The trained neural network model.
    """
    #Fix randomness 
    tf.random.set_seed(675)
    
    
    # More complex neural network model
    model = Sequential([
        Dense(2048, input_shape=(input_shape,)),
        Activation('relu'),
        Dense(1024),
        Activation('relu'),
        Dense(512),
        Activation('relu'),
        Dense(256),
        Activation('relu'),
        Dense(128),
        Activation('relu'),
        Dense(64),
        Activation('relu'),
        Dense(32),
        Activation('relu'),
        Dense(16),
        Activation('relu'),
        Dense(8),
        Activation('relu'),
        Dense(4),
        Activation('relu'),
        Dense(2),
        Activation('relu'),
        Dense(1)
    ])
    # Compile the model with an initial learning rate
    initial_learning_rate = initial_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(scheduler)

    # Fit the model with the learning rate scheduler
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler], validation_split=0.2)
    
    # Calculate the Mean Absolute Error on the test set
    return model
    
def NLP_depth_10(X_train, y_train, input_shape,initial_rate=0.001,batch_size = 8,epochs=80):
    """
    Create and train a more complex neural network model for Natural Language Processing (NLP) tasks.

    This function defines a multi-layer neural network model with several hidden layers, ReLU activation functions,
    and uses mean squared error as the loss function. It also includes a learning rate scheduler callback to adjust
    the learning rate during training.

    Parameters:
        X_train (numpy.ndarray): Input training data.
        y_train (numpy.ndarray): Target training data.
        input_shape (int): The shape of the input data.

    Returns:
        tensorflow.keras.models.Sequential: The trained neural network model.
    """
    
    #Fix randomness 
    tf.random.set_seed(675)
    
    
    # More complex neural network model
    model = Sequential([
        Dense(2048, input_shape=(input_shape,)),
        Activation('relu'),
        Dense(1024),
        Activation('relu'),
        Dense(512),
        Activation('relu'),
        Dense(256),
        Activation('relu'),
        Dense(128),
        Activation('relu'),
        Dense(64),
        Activation('relu'),
        Dense(32),
        Activation('relu'),
        Dense(16),
        Activation('relu'),
        Dense(8),
        Activation('relu'),
        Dense(4),
        Activation('relu'),
        Dense(1)
    ])
    # Compile the model with an initial learning rate
    initial_learning_rate = initial_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(scheduler)

    # Fit the model with the learning rate scheduler
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler], validation_split=0.2)
    
    # Calculate the Mean Absolute Error on the test set
    return model





def NLP_depth_9(X_train, y_train, input_shape,initial_rate=0.001,batch_size = 8,epochs=80):
    """
    Create and train a more complex neural network model for Natural Language Processing (NLP) tasks.

    This function defines a multi-layer neural network model with several hidden layers, ReLU activation functions,
    and uses mean squared error as the loss function. It also includes a learning rate scheduler callback to adjust
    the learning rate during training.

    Parameters:
        X_train (numpy.ndarray): Input training data.
        y_train (numpy.ndarray): Target training data.
        input_shape (int): The shape of the input data.

    Returns:
        tensorflow.keras.models.Sequential: The trained neural network model.
    """
    
    #Fix randomness 
    tf.random.set_seed(675)
    
    
    # More complex neural network model
    model = Sequential([
        Dense(2048, input_shape=(input_shape,)),
        Activation('relu'),
        Dense(1024),
        Activation('relu'),
        Dense(512),
        Activation('relu'),
        Dense(256),
        Activation('relu'),
        Dense(128),
        Activation('relu'),
        Dense(64),
        Activation('relu'),
        Dense(32),
        Activation('relu'),
        Dense(16),
        Activation('relu'),
        Dense(8),
        Activation('relu'),
        Dense(1)
    ])

    # Compile the model with an initial learning rate
    initial_learning_rate = initial_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(scheduler)

    # Fit the model with the learning rate scheduler
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler], validation_split=0.2)
    
    # Calculate the Mean Absolute Error on the test set
    return model

def NLP_depth_8(X_train, y_train, input_shape,initial_rate=0.001,batch_size = 8,epochs=80):
    """
    Create and train a more complex neural network model for Natural Language Processing (NLP) tasks.

    This function defines a multi-layer neural network model with several hidden layers, ReLU activation functions,
    and uses mean squared error as the loss function. It also includes a learning rate scheduler callback to adjust
    the learning rate during training.

    Parameters:
        X_train (numpy.ndarray): Input training data.
        y_train (numpy.ndarray): Target training data.
        input_shape (int): The shape of the input data.

    Returns:
        tensorflow.keras.models.Sequential: The trained neural network model.
    """
    #Fix randomness 
    tf.random.set_seed(675)
    
    
    # More complex neural network model
    model = Sequential([
        Dense(2048, input_shape=(input_shape,)),
        Activation('relu'),
        Dense(1024),
        Activation('relu'),
        Dense(512),
        Activation('relu'),
        Dense(256),
        Activation('relu'),
        Dense(128),
        Activation('relu'),
        Dense(64),
        Activation('relu'),
        Dense(32),
        Activation('relu'),
        Dense(16),
        Activation('relu'),
        Dense(1)
    ])

    # Compile the model with an initial learning rate
    initial_learning_rate = initial_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(scheduler)

    # Fit the model with the learning rate scheduler
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler], validation_split=0.2)
    
    # Calculate the Mean Absolute Error on the test set
    return model


# In[2]:


def NLP_depth_7(X_train, y_train, input_shape,initial_rate=0.001,batch_size = 8,epochs=80):
    """
    Create and train a more complex neural network model for Natural Language Processing (NLP) tasks.

    This function defines a multi-layer neural network model with several hidden layers, ReLU activation functions,
    and uses mean squared error as the loss function. It also includes a learning rate scheduler callback to adjust
    the learning rate during training.

    Parameters:
        X_train (numpy.ndarray): Input training data.
        y_train (numpy.ndarray): Target training data.
        input_shape (int): The shape of the input data.

    Returns:
        tensorflow.keras.models.Sequential: The trained neural network model.
    """
    
    #Fix randomness 
    tf.random.set_seed(675)
    
    # More complex neural network model
    model = Sequential([
        Dense(2048, input_shape=(input_shape,)),
        Activation('relu'),
        Dense(1024),
        Activation('relu'),
        Dense(512),
        Activation('relu'),
        Dense(256),
        Activation('relu'),
        Dense(128),
        Activation('relu'),
        Dense(64),
        Activation('relu'),
        Dense(32),
        Activation('relu'),
        Dense(1)
    ])

    # Compile the model with an initial learning rate
    initial_learning_rate = initial_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(scheduler)

    # Fit the model with the learning rate scheduler
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler], validation_split=0.2)
    
    # Calculate the Mean Absolute Error on the test set
    return model


# In[3]:


def NLP_depth_5(X_train, y_train, input_shape,initial_rate=0.001,batch_size = 8,epochs=80):
    """
    Create and train a more complex neural network model for Natural Language Processing (NLP) tasks.

    This function defines a multi-layer neural network model with several hidden layers, ReLU activation functions,
    and uses mean squared error as the loss function. It also includes a learning rate scheduler callback to adjust
    the learning rate during training.

    Parameters:
        X_train (numpy.ndarray): Input training data.
        y_train (numpy.ndarray): Target training data.
        input_shape (int): The shape of the input data.

    Returns:
        tensorflow.keras.models.Sequential: The trained neural network model.
    """
    
    #Fix randomness 
    tf.random.set_seed(675)
    
    
    # More complex neural network model
    model = Sequential([
        Dense(2048, input_shape=(input_shape,)),
        Activation('relu'),
        Dense(1024),
        Activation('relu'),
        Dense(512),
        Activation('relu'),
        Dense(256),
        Activation('relu'),
        Dense(128),
        Activation('relu'),
        Dense(1)
    ])

    # Compile the model with an initial learning rate
    initial_learning_rate = initial_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(scheduler)

    # Fit the model with the learning rate scheduler
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler], validation_split=0.2)
    
    # Calculate the Mean Absolute Error on the test set
    return model


# In[4]:


def NLP_depth_3(X_train, y_train, input_shape,initial_rate=0.001,batch_size = 8,epochs=80):
    """
    Create and train a more complex neural network model for Natural Language Processing (NLP) tasks.

    This function defines a multi-layer neural network model with several hidden layers, ReLU activation functions,
    and uses mean squared error as the loss function. It also includes a learning rate scheduler callback to adjust
    the learning rate during training.

    Parameters:
        X_train (numpy.ndarray): Input training data.
        y_train (numpy.ndarray): Target training data.
        input_shape (int): The shape of the input data.

    Returns:
        tensorflow.keras.models.Sequential: The trained neural network model.
    """
    
    #Fix randomness 
    tf.random.set_seed(675)
    
    
    # More complex neural network model
    model = Sequential([
        Dense(2048, input_shape=(input_shape,)),
        Activation('relu'),
        Dense(1024),
        Activation('relu'),
        Dense(512),
        Activation('relu'),
        Dense(1)
    ])

    # Compile the model with an initial learning rate
    initial_learning_rate = initial_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(scheduler)

    # Fit the model with the learning rate scheduler
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler], validation_split=0.2)
    
    # Calculate the Mean Absolute Error on the test set
    return model


# In[5]:


def NLP_depth_1(X_train, y_train, input_shape,initial_rate=0.001,batch_size = 8,epochs=80):
    """
    Create and train a more complex neural network model for Natural Language Processing (NLP) tasks.

    This function defines a multi-layer neural network model with several hidden layers, ReLU activation functions,
    and uses mean squared error as the loss function. It also includes a learning rate scheduler callback to adjust
    the learning rate during training.

    Parameters:
        X_train (numpy.ndarray): Input training data.
        y_train (numpy.ndarray): Target training data.
        input_shape (int): The shape of the input data.

    Returns:
        tensorflow.keras.models.Sequential: The trained neural network model.
    """
    
    #Fix randomness 
    tf.random.set_seed(675)
    
    
    
    # More complex neural network model
    model = Sequential([
        Dense(16384, input_shape=(input_shape,)),
        Activation('relu'),
        Dense(1)
    ])

    # Compile the model with an initial learning rate
    initial_learning_rate = initial_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Learning rate scheduler callback
    lr_scheduler = LearningRateScheduler(scheduler)

    # Fit the model with the learning rate scheduler
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler], validation_split=0.2)
    
    # Calculate the Mean Absolute Error on the test set
    return model


# In[ ]:





# In[ ]:





# In[ ]:




