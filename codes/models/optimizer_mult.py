from .clr import cyclic_learning_rate
import tensorflow.compat.v1 as tf
import numpy as np

class Optimizer():
    def __init__(self, model, preds, d_labels, lr, d_pos_weights, d_indexs):

        global_step = tf.Variable(0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=cyclic_learning_rate(global_step=global_step, learning_rate=lr*0.1, max_lr=lr, mode='exp_range', gamma=.995))

        self.cost = 0
        for idx in range(len(d_labels)):
            each_preds = tf.gather(tf.reshape(preds[idx],[-1,1]), d_indexs[idx])
            each_labels = np.array(d_labels[idx], dtype=np.float32).reshape(-1,1)
            each_pos_weight = d_pos_weights[idx]
            each_cost = tf.reduce_mean( tf.nn.weighted_cross_entropy_with_logits(logits=each_preds, targets=each_labels, pos_weight=each_pos_weight) / each_labels.shape[0] ) 
            self.cost += each_cost

        self.opt_op = self.optimizer.minimize(self.cost, global_step=global_step)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
