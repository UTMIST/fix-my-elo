"""
Two-headed CNN for picking the next move best move (policy) and 
estimating win probability (value).

=====================================
LOGIC REFERENCED FROM ALPHAZERO PAPER
=====================================


"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
import config

def build_residual_block(x, filters=256):
    # save residual
    shortcut = x 
    
    # looks for <filters> # of patterns in 3x3 chunks of the board
    x = Conv2D(filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x) # normalize to mean 0 std 1
    x = ReLU()(x)
    # TODO: could drop
    # convolute input further to extract more complex features
    x = Conv2D(filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)

    # add the original input (residual) to the processed output
    x = Add()([x, shortcut]) 
    x = ReLU()(x)
    return x

def build_model():
    inputs = Input(shape=config.INPUT_SHAPE, name='board_input')

    # initial Convolution Block
    x = Conv2D(256, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # build several residual blocks ("residual tower")
    for _ in range(4):
        x = build_residual_block(x)

    # POLICY NETWORK HEAD ==> probability distribution for best next move
    p = Conv2D(2, kernel_size=1)(x) # 1x1 Conv to reduce depth
    p = BatchNormalization()(p)
    p = ReLU()(p)
    p = Flatten()(p)
    policy_out = Dense(config.ACTION_SPACE_SIZE, activation='softmax', name='policy')(p)

    # VALUE NETWORK HEAD ==> WIN/LOSS PREDICTION (-1 to 1)
    v = Conv2D(1, kernel_size=1)(x)
    v = BatchNormalization()(v)
    v = ReLU()(v)
    v = Flatten()(v)
    v = Dense(256, activation='relu')(v)
    value_out = Dense(1, activation='tanh', name='value')(v)

    # final model (use cross-entropy for policy, MSE for value network loss functions)
    model = Model(inputs=inputs, outputs=[policy_out, value_out])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
        loss={'policy': 'categorical_crossentropy', 'value': 'mean_squared_error'}
    )
    
    return model
