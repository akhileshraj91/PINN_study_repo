import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


N = 200
X = np.random.random(N)


sign = (- np.ones((N,)))**np.random.randint(2,size=N)
Y = np.sqrt(X) * sign


act = tf.keras.layers.ReLU()
nn_sv = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation=act),
     tf.keras.layers.Dense(10, activation=act),
      tf.keras.layers.Dense(1,activation='linear')])

mse = tf.keras.losses.MeanSquaredError()
def loss_dp(y_true, y_pred):
    return mse(y_true,y_pred**2)
# loss_sv = tf.keras.losses.MeanSquaredError()
optimizer_sv = tf.keras.optimizers.Adam(lr=0.001)
nn_sv.compile(optimizer=optimizer_sv, loss=loss_dp)
# Training
results_sv = nn_sv.fit(X, X, epochs=5, batch_size= 5, verbose=1)

plt.plot(X,Y,'.',label='Data points', color="lightgray") 
plt.plot(X,nn_sv.predict(X),'.',label='Supervised', color="red")
plt.xlabel('y')
plt.ylabel('x')
plt.title('Standard approach') 
plt.legend()
plt.show()