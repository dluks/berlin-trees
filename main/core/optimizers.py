import tensorflow as tf

adaDelta = tf.optimizers.Adadelta(
    learning_rate=1.0, rho=0.95, weight_decay=None
)
adam = tf.optimizers.Adam(
    learning_rate=5.0e-5, beta_1=0.9, beta_2=0.999, epsilon=1.0e-8, weight_decay=0.0
)
nadam = tf.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=None)
adagrad = tf.optimizers.Adagrad(learning_rate=0.01, epsilon=None, weight_decay=0.0)
