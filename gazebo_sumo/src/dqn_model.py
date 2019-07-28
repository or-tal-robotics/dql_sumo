import tensorflow as tf
import numpy as np

class DQN():
    def __init__(self, K, conv_layer_sizes, hidden_layer_sizes, scope, image_size):
        self.K = K
        self.scope = scope
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.X = tf.placeholder(tf.float32, shape=(None, image_size,image_size, 4), name='X')
            self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
            self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
            Z = self.X / 255.0
            for num_output_filters, filtersz, poolsz in conv_layer_sizes:
                Z = tf.contrib.layers.conv2d(Z, num_output_filters, filtersz, poolsz, activation_fn=tf.nn.relu)
            
            Z = tf.contrib.layers.flatten(Z)
            for M in hidden_layer_sizes:
                Z = tf.contrib.layers.fully_connected(Z,M)
            self.predict_op = tf.contrib.layers.fully_connected(Z,K)
            selected_action_value = tf.reduce_sum(self.predict_op * tf.one_hot(self.actions,K), reduction_indices=[1])
            
            cost = tf.reduce_mean(tf.losses.huber_loss(self.G, selected_action_value))
            self.train_op = tf.train.AdamOptimizer(1e-5).minimize(cost)
            self.cost = cost
            
    def copy_from(self, other):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key=lambda v: v.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        theirs = sorted(theirs, key=lambda v: v.name)
        
        ops = []
        for p,q in zip(mine, theirs):
            op = p.assign(q)
            ops.append(op)
        self.session.run(ops)
    
    def save(self):
        params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        params = self.session.run(params)
        np.savez('tf_dqn_weights.npz', *params)
    
    def load(self):
        params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        npz = np.load('tf_dqn_weights.npz')
        ops = []
        for p, (_, v) in zip(params, npz.iteritems()):
            ops.append(p.assign(v))
        self.session.run(ops)
        
    def set_session(self,session):
        self.session = session
    
    def predict(self, states):
        return self.session.run(self.predict_op, feed_dict = {self.X: states})
    
    def update(self, states, actions, targets):
        c, _ = self.session.run(
                [self.cost, self.train_op],
                feed_dict = {self.X: states, self.G: targets, self.actions: actions}
                )
        return c
    
    def sample_action(self,x,eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            return np.argmax(self.predict([x])[0])