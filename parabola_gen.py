import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


N = 200
X = np.random.random(N)

sign = (- np.ones((N)))**np.random.randint(2,size=N)
Y = np.sqrt(X) * sign


act = tf.keras.layers.ReLU()

nn_dp = tf.keras.models.Sequential([
  tf.keras.layers.Dense(10, activation=act),
  tf.keras.layers.Dense(10, activation=act),
  tf.keras.layers.Dense(1, activation='linear')])

mse = tf.keras.losses.MeanSquaredError()
def loss_dp(y_true, y_pred):
    return mse(y_true,y_pred**2)

optimizer_dp = tf.keras.optimizers.Adam(lr=0.001)
nn_dp.compile(optimizer=optimizer_dp, loss=loss_dp)

#Training
results_dp = nn_dp.fit(X, X, epochs=5, batch_size=5, verbose=1)


# Results
plt.figure(2)
plt.plot(X,Y,'.',label='Datapoints', color="lightgray")
#plt.plot(X,nn_sv.predict(X),'.',label='Supervised', color="red") # optional for comparison
plt.plot(X,nn_dp.predict(X),'.',label='Diff. Phys.', color="green") 
plt.xlabel('x')
plt.ylabel('y')
plt.title('Differentiable physics approach')
plt.legend()
plt.show()