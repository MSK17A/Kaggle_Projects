import tensorflow as tf

def My_Model(input_shape=(7)):

    """
    x               Input data
    input_shape     (7,1) 
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(input_shape))
    model.add(tf.keras.layers.Dense(units=20, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(units=20, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(units=1))
    
    return model
