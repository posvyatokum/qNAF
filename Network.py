import tensorflow as tf
import tflearn
import numpy as np 
from tensorflow.contrib.framework import get_variables

def get_triu_with_exp_diag(l, a_dim):
    pivot = 0
    rows = []
    for idx in range(a_dim):
        count = a_dim - idx
        diag_elem = tf.exp(tf.slice(l, (0, pivot), (-1, 1)))
        non_diag_elems = tf.slice(l, (0, pivot+1), (-1, count-1))
        row = tf.pad(tf.concat(axis=1, values=[diag_elem, non_diag_elems]), ((0, 0), (idx, 0)))
        rows.append(row)
        pivot += count
    return tf.transpose(tf.stack(rows, axis=1), (0, 2, 1))

class Network:
    def __init__(self, sess, state_dim, action_dim, learning_rate, num_prev_params, scope='NAF', sigma_P_dep=False, det=True, hn=0):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.a_bound = 1.0
        self.learning_rate = learning_rate
        self.h_const = tf.cast(0.5 * (np.log(2 * np.pi) + 1), tf.float32)
    
        with tf.variable_scope(scope + 'common'):
            self.inputs_x = tf.placeholder(tf.float32, shape=[None, self.s_dim])
            self.inputs_u = tf.placeholder(tf.float32, shape=[None, self.a_dim])
            self.hidden1 = tf.contrib.layers.fully_connected(
                           self.inputs_x, 200, activation_fn=tf.nn.relu)
            self.hidden4 = tf.contrib.layers.fully_connected(
                           self.hidden1, 200, activation_fn=tf.nn.relu)

        with tf.variable_scope(scope + 'mu_ub'):
            self.mu_ub = tf.contrib.layers.fully_connected(
                     self.hidden4, self.a_dim, activation_fn=tf.nn.tanh)
            self.mu_ub = tf.reshape(self.mu_ub, [-1, self.a_dim])

        with tf.variable_scope(scope + 'mu_det'):
            self.mu_det = tf.clip_by_value(self.mu_ub, -self.a_bound, self.a_bound)
                
        with tf.variable_scope(scope + 'V'):
            self.V = tf.contrib.layers.fully_connected(
                     self.hidden4, 1, activation_fn=None)
            self.V = tf.reshape(self.V, [-1, 1])

        with tf.variable_scope(scope + 'P'):
            self.L = tf.contrib.layers.fully_connected(
                     self.hidden4, (self.a_dim * (self.a_dim + 1)) // 2, activation_fn=None)
            self.L = tf.reshape(self.L, [-1, (self.a_dim * (self.a_dim + 1)) // 2])

            self.L_triu = get_triu_with_exp_diag(self.L, self.a_dim)

            self.P = tf.matmul(self.L_triu, tf.transpose(self.L_triu, (0, 2, 1)))
            self.P = tf.add(self.P, tf.multiply(1e-9, tf.eye(self.a_dim)))

        with tf.variable_scope(scope + 'mu_norm'):
            #TODO s_dim > 1
            if sigma_P_dep:
                self.P_inv = tf.matrix_inverse(self.P)
                if hn == 0:
                    self.sigma = tflearn.fully_connected(
                                 self.P_inv, 1)
                    self.C = self.sigma.W
                else: 
                    self.hidden_sigma = tf.contrib.layers.fully_connected(
                    self.P_inv, hn, activation_fn=tf.nn.relu)
                    self.sigma = tf.contrib.layers.fully_connected(
                        self.hidden_sigma, 1, activation_fn=tf.nn.relu)

            else:
                self.sigma = tf.contrib.layers.fully_connected(
                             self.hidden4, 1, activation_fn=None)
            self.sigma = tf.reshape(self.sigma, [-1, 1])
            self.sigma = tf.abs(self.sigma)
            self.pi_normal = tf.contrib.distributions.Normal(self.mu_ub, self.sigma) # * noise const
            self.mu_norm = self.pi_normal.sample(1)
            self.mu_norm = tf.reshape(self.mu_norm, (-1, self.a_dim))
            self.mu_norm = tf.clip_by_value(self.mu_norm, -self.a_bound, self.a_bound)
            self.log_prob = self.pi_normal.log_prob(self.inputs_u)

        with tf.variable_scope(scope + 'A'):
            if det:
                tmp = tf.expand_dims(self.inputs_u - self.mu_det, -1)
            else:
                tmp = tf.expand_dims(self.inputs_u - self.mu_norm, -1)
            self.A = -tf.matmul(tf.transpose(tmp, [0, 2, 1]), tf.matmul(self.P, tmp))/2
            self.A = tf.reshape(self.A, [-1, 1])

        with tf.variable_scope(scope + 'Q'):
            self.Q = self.A + self.V

        self.mu_norm_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + 'common') +\
                              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + 'mu_norm') +\
                              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + 'mu_ub')
        if sigma_P_dep:
            self.mu_norm_params = self.mu_norm_params +\
                                  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + 'P') 
    
            ### loss
        with tf.variable_scope(scope + 'loss'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.inputs_y = tf.placeholder(shape=[None, 1],dtype=tf.float32)

            self.td_error = tf.square(self.inputs_y - self.Q)
            self.loss = tf.reduce_mean(self.td_error)
            self.update_model = self.optimizer.minimize(self.loss)

        with tf.variable_scope(scope + 'loss_spg'):
            #TODO s_dim > 1
            self.inputs_Q = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            self.log_probs = -0.5 * tf.div(tf.pow(self.inputs_x - self.mu_ub, 2), tf.pow(self.sigma, 2))
            self.log_probs = self.log_probs - tf.pow(2 * np.pi * tf.pow(self.sigma, 2), 0.5)

            self.loss_spg = -tf.reduce_mean(tf.multiply(self.log_probs, self.inputs_Q))
            self.optimizer_spg = tf.train.AdamOptimizer(self.learning_rate)
            self.optimize_spg = self.optimizer_spg.minimize(self.loss_spg)

        with tf.variable_scope(scope + 'Vloss'):
            self.V_sep = tf.contrib.layers.fully_connected(
                     self.hidden4, 1, activation_fn=None)
            self.V_sep = tf.reshape(self.V_sep, [-1, 1])
            self.optimizer_V = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.inputs_yV = tf.placeholder(shape=[None, 1],dtype=tf.float32)

            self.td_error_V = tf.square(self.inputs_yV - self.V_sep)
            self.loss_V = tf.reduce_mean(self.td_error_V)
            self.update_model_V = self.optimizer_V.minimize(self.loss_V)

        self.variables = tf.trainable_variables()[num_prev_params: ] 
                            
    def predict_u_det(self, inputs_x):
        return self.sess.run(self.mu_det, 
                             feed_dict={self.inputs_x: inputs_x.reshape(-1, self.s_dim)})

    def predict_u_norm(self, inputs_x):
        return self.sess.run(self.mu_norm, 
                             feed_dict={self.inputs_x: inputs_x.reshape(-1, self.s_dim)})

    def predict_V(self, inputs_x):
        return self.sess.run(self.V, 
                             feed_dict={self.inputs_x: inputs_x.reshape(-1, self.s_dim)})

    def predict_V_sep(self, inputs_x):
        return self.sess.run(self.V_sep, 
                             feed_dict={self.inputs_x: inputs_x.reshape(-1, self.s_dim)})

    def predict_Q(self, inputs_x, inputs_u):
        return self.sess.run(self.V, 
                             feed_dict={self.inputs_x: inputs_x.reshape(-1, self.s_dim),
                                        self.inputs_u: inputs_u.reshape(-1, self.a_dim)})

    def get_log_prob(self, inputs_x, inputs_u):
        return self.sess.run(self.log_prob,
                             feed_dict={self.inputs_x: inputs_x.reshape(1, self.s_dim),
                                        self.inputs_u: inputs_u.reshape(1, self.a_dim)})

    def get_log_probs(self, inputs_x, inputs_u):
        return self.sess.run(self.log_probs,
                             feed_dict={self.inputs_x: inputs_x.reshape(-1, self.s_dim),
                                        self.inputs_u: inputs_u.reshape(-1, self.a_dim)}) 

    def update_Q(self, inputs_x, inputs_u, inputs_y):
        return self.sess.run(self.update_model_V, 
                             feed_dict={self.inputs_x: inputs_x.reshape(-1, self.s_dim),
                                        self.inputs_u: inputs_u.reshape(-1, self.a_dim), 
                                        self.inputs_y: inputs_y.reshape(-1, 1)})

    def update_V_sep(self, inputs_x, inputs_y):
        return self.sess.run(self.update_model_V, 
                             feed_dict={self.inputs_x: inputs_x.reshape(-1, self.s_dim),
                                        self.inputs_yV: inputs_y.reshape(-1, 1)})

    def update_Q(self, inputs_x, inputs_u, inputs_y):
        return self.sess.run(self.update_model, 
                             feed_dict={self.inputs_x: inputs_x.reshape(-1, self.s_dim),
                                        self.inputs_u: inputs_u.reshape(-1, self.a_dim), 
                                        self.inputs_y: inputs_y.reshape(-1, 1)})
    def update_mu(self, inputs_x, inputs_u, inputs_Q):
        return self.sess.run(self.optimize_spg,
                             feed_dict={
                                    self.inputs_x: inputs_x.reshape(-1, self.s_dim),
                                    self.inputs_u: inputs_u.reshape(-1, self.a_dim),
                                    self.inputs_Q: inputs_Q.reshape(-1, 1),
                             })

    def make_soft_update_from(self, network, tau):
        assert len(network.variables) == len(self.variables), \
          "target and prediction network should have same # of variables"

        self.assign_op = {}
        for from_, to_ in zip(network.variables, self.variables):
            if 'BatchNorm' in to_.name:
                self.assign_op[to_.name] = to_.assign(from_)
            else:
                self.assign_op[to_.name] = to_.assign(tau * from_ + (1-tau) * to_)

    def hard_copy_from(self, network):
        assert len(network.variables) == len(self.variables), \
          "target and prediction network should have same # of variables"

        for from_, to_ in zip(network.variables, self.variables):
            self.sess.run(to_.assign(from_))

    def soft_update_from(self, network):
        for variable in self.variables:
            self.sess.run(self.assign_op[variable.name])
        return True
