import tensorflow as tf
import numpy as np
from tensorflow import keras
def house_model(y_new):
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=float)
    ys = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 450.0], dtype=float)
    
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd',loss='mean_squared_error')
    model.fit(xs, ys, epochs=4000)
    
    return (model.predict(y_new)[0]+1) //100
    
# Function call
prediction = house_model([10.0])
print(prediction)
