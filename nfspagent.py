import tensorflow as tf


class NFSPAgent(object):

    def __init__(self, input_number, hidden_units, logger, learning_rate, actions):
        self.logger = logger
        self.actions = actions

        self.prediction_scope = self.name + '-prediction_network'
        self.pred_states, self.pred_q_values = self.build_network(self.prediction_scope, input_number, hidden_units)

        if learning_rate is not None:
            self.train_target_q, self.train_actions, self.train_loss, self.train_operation = self.build_training_operation(
                learning_rate)

    def build_network(self, variable_scope, input_number, hidden_units):
        with tf.variable_scope(variable_scope):
            states = tf.placeholder(tf.float32, shape=[None, input_number], name="state")

            initializer = tf.variance_scaling_initializer()
            hidden_layer = tf.layers.dense(states, hidden_units, activation=tf.nn.elu, kernel_initializer=initializer,
                                           name="hidden")
            outputs = tf.layers.dense(hidden_layer, len(self.actions), kernel_initializer=initializer, name="q_values")

        return states, outputs

    def build_training_operation(self, learning_rate):
        train_target_q = tf.placeholder(tf.float32, [None], name="target_q_values")
        train_actions = tf.placeholder(tf.int64, [None], name="actions")

        actions_one_hot = tf.one_hot(train_actions, len(self.actions), 1.0, 0.0, name="actions_one_hot")
        action_q_values = tf.reduce_sum(self.pred_q_values * actions_one_hot, axis=1, name="action_q_values")

        delta = tf.square(train_target_q - action_q_values)
        loss = tf.reduce_mean(delta, name="loss")
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        return train_target_q, train_actions, loss, optimizer.minimize(loss)


def main():
    print("Start!")


if __name__ == "__main__":
    main()
