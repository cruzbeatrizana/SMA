import random
from collections import deque

from tensorflow.keras.models import model_from_json, load_model, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from pyglet.window import key

# individual agent policy


class Policy(object):
    def __init__(self):
        pass

    def action(self, obs):
        raise NotImplementedError()

# interactive policy based on keyboard input
# hard-coded to deal only with movement, not communication


class InteractivePolicy(Policy):
    def __init__(self, env, agent_index):
        super(InteractivePolicy, self).__init__()
        self.env = env
        # hard-coded keyboard events
        self.move = [False for i in range(4)]
        self.comm = [False for i in range(env.world.dim_c)]
        # register keyboard events with this environment's window
        env.viewers[agent_index].window.on_key_press = self.key_press
        env.viewers[agent_index].window.on_key_release = self.key_release

    def action(self, obs):
        # ignore observation and just act based on keyboard events
        if self.env.discrete_action_input:
            u = 0
            if self.move[0]:
                u = 1
            if self.move[1]:
                u = 2
            if self.move[2]:
                u = 4
            if self.move[3]:
                u = 3
        else:
            u = np.zeros(5)  # 5-d because of no-move action
            if self.move[0]:
                u[1] += 1.0
            if self.move[1]:
                u[2] += 1.0
            if self.move[3]:
                u[3] += 1.0
            if self.move[2]:
                u[4] += 1.0
            if True not in self.move:
                u[0] += 1.0
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

    # keyboard event callbacks
    def key_press(self, k, mod):
        if k == key.LEFT:
            self.move[0] = True
        if k == key.RIGHT:
            self.move[1] = True
        if k == key.UP:
            self.move[2] = True
        if k == key.DOWN:
            self.move[3] = True

    def key_release(self, k, mod):
        if k == key.LEFT:
            self.move[0] = False
        if k == key.RIGHT:
            self.move[1] = False
        if k == key.UP:
            self.move[2] = False
        if k == key.DOWN:
            self.move[3] = False


class Navigation(Policy):

    def __init__(self, env, batch_size=256, epochs=16, lr=1e-4, gamma=0.99, memory=128, epsilon=0.9, import_model=False, num=None, run_logdir=None, d_dense=128, alpha=0.9):
        super(Navigation, self).__init__()

        # self.sess = keras.backend.get_session()

        self.env = env
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.memory = memory
        self.queue = deque([], maxlen=self.memory)
        self.epsilon = epsilon
        self.run_logdir = run_logdir
        self.train_summary_writer = tf.summary.create_file_writer(
            self.run_logdir)
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_accuracy = tf.keras.metrics.Accuracy(
            'train_accuracy')
        self.train_count = 0
        self.clipping_loss_ratio = 0.2
        self.entropy_loss_ratio = 5e-3
        self.d_dense = d_dense
        self.alpha = alpha

        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)

        self.input_shape = self.env.observation_space[0].shape
        self.output_shape = self.env.action_space[0].n

        self.dummy_advantage = np.zeros((1, 1))
        self.dummy_old_prediction = np.zeros((1, self.output_shape))

        # tf.compat.v1.disable_eager_execution()

        # Actor Network (S) -> pi(s) = [pi(1), pi(a2), pi(3), ...]
        actor_state = Input(
            shape=self.input_shape, name="state")
        advantage = Input(shape=(1, ), name="Advantage")
        old_prediction = Input(
            shape=(self.output_shape,), name="Old_Prediction")

        actor_shared_hidden = self._shared_network_structure(actor_state)

        policy = Dense(self.output_shape, activation="softmax",
                       name="actor_output_layer")(actor_shared_hidden)

        self.actor_network = Model(
            inputs=[actor_state, advantage, old_prediction], outputs=policy)
        # self.actor_network.compile(optimizer=self.optimizer)
        self.actor_old_network = self.build_network_from_copy(
            self.actor_network)

        # Critic Network (S) -> V(s') = Q(s') = R + gamma * V(s)
        critic_state = Input(
            shape=self.input_shape, name="state")
        critic_shared_hidden = self._shared_network_structure(critic_state)

        q = Dense(1, name="critic_output_layer")(critic_shared_hidden)

        self.critic_network = Model(inputs=critic_state, outputs=q)

        self.critic_network.compile(
            optimizer=tf.keras.optimizers.Adam(lr=self.lr), loss="mse")

        # Input = Observation space
        # if import_model:
        #    self.model = tf.keras.models.load_model(
        #        "model_agent" + str(num) + ".h5")
        #    self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        # else:
        #    self.model = tf.keras.models.Sequential([
        #        tf.keras.layers.Dense(16, input_shape=self.input_shape),  # fc
        #        tf.keras.layers.BatchNormalization(),  # norm
        #        tf.keras.layers.Dense(32, activation="relu"),  # relu
        #        tf.keras.layers.Dense(64),  # fc
        #        tf.keras.layers.BatchNormalization(),  # norm
        #        tf.keras.layers.Dense(32, activation="relu"),  # relu
        #        tf.keras.layers.Dense(16),  # fc
        #        tf.keras.layers.BatchNormalization(),  # norm
        #        tf.keras.layers.Dense(8, activation="relu"),  # relu
        #        tf.keras.layers.Dense(self.output_shape)
        #    ])
        #    self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)

    def _shared_network_structure(self, state_features):
        dense_d = self.d_dense
        hidden1 = tf.keras.layers.Dense(dense_d, activation="tanh",
                                        name="hidden_shared_1")(state_features)
        hidden2 = tf.keras.layers.Dense(dense_d, activation="tanh",
                                        name="hidden_shared_2")(hidden1)
        return hidden2

    def build_network_from_copy(self, actor_network):
        network_structure = actor_network.to_json()
        network_weights = actor_network.get_weights()
        network = model_from_json(network_structure)
        network.set_weights(network_weights)
        network.compile(optimizer=tf.keras.optimizers.Adam(
            lr=self.lr), loss="mse")
        return network

    # def proximal_policy_optimization_loss(self, advantage, old_prediction):
    #     loss_clipping = self.clipping_loss_ratio
    #     entropy_loss = self.entropy_loss_ratio
#
    #     def loss(y_true, y_pred):
    #         prob = y_true * y_pred
    #         old_prob = y_true * old_prediction
    #         r = prob / (old_prob + 1e-10)
    #         return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - loss_clipping,
    #                                                        max_value=1 + loss_clipping) * advantage) + entropy_loss * (
    #                        prob * K.log(prob + 1e-10)))
#
    #     return loss

    def loss(self, y_true, y_pred, advantage, old_prediction):
        loss_clipping = self.clipping_loss_ratio
        entropy_loss = self.entropy_loss_ratio
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        r = prob / (old_prob + 1e-10)
        return -K.mean(np.minimum(r * advantage, K.clip(r, min_value=1 - loss_clipping,
                                                        max_value=1 + loss_clipping) * advantage) + entropy_loss * (
            prob * K.log(prob + 1e-10)))

    def action(self, state):

        assert isinstance(state, np.ndarray), "state must be numpy.ndarry"

        state = np.reshape(state, [-1, self.input_shape[0]])
        #print(state.shape, state)
        prob = self.actor_network.predict_on_batch(
            [state, self.dummy_advantage, self.dummy_old_prediction]).flatten()
        action = np.random.choice(self.output_shape, p=prob)
        u = tf.one_hot(action, self.output_shape)

        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

    def train(self):

        states, actions, rewards, next_states, dones = self.sample_experiences(
            self.batch_size)
        discounted_r = []

        if dones[-1]:
            v = 0
        else:
            v = self.get_v(next_states[-1])
        for r in rewards[::-1]:
            v = r + self.gamma * v
            discounted_r.append(v)
        discounted_r.reverse()

        batch_s, batch_a, batch_discounted_r = np.vstack(states), np.vstack(
            actions), np.vstack(discounted_r)

        batch_v = self.get_v(batch_s)
        batch_advantage = batch_discounted_r - batch_v
        batch_old_prediction = self.get_old_prediction(batch_s)

        batch_a_final = tf.one_hot(batch_a.flatten(), depth=self.output_shape)

        # batch_a_final = np.zeros(shape=(len(batch_a), self.output_shape))
        # batch_a_final[:, batch_a.flatten()] = 1

        # self.actor_network.fit(x=[batch_s, batch_advantage, batch_old_prediction], y=batch_a_final, verbose=0)

        for _ in range(10):
            with tf.GradientTape() as tape:
                y_pred = self.actor_network(
                    [batch_s, batch_advantage, batch_old_prediction])
                # print(Q_values, target_Q_values)
                loss = self.loss(batch_a_final, y_pred,
                                 batch_advantage, batch_old_prediction)

            grads = tape.gradient(loss, self.actor_network.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.actor_network.trainable_variables))

            self.update_target_network()

        history = self.critic_network.fit(
            x=batch_s, y=batch_discounted_r, epochs=10, verbose=0)

        with self.train_summary_writer.as_default():
            tf.summary.scalar("Actor_loss", loss, step=self.train_count)
            tf.summary.scalar(
                "Critic_loss", np.average(history.history["loss"]), step=self.train_count)

        self.train_count += 1

        self.queue.clear()

    def get_v(self, s):
        s = np.reshape(s, (-1, self.input_shape[0]))
        v = self.critic_network.predict_on_batch(s)
        return v

    def update_target_network(self):
        alpha = self.alpha
        self.actor_old_network.set_weights(alpha*np.array(self.actor_network.get_weights())
                                           + (1-alpha)*np.array(self.actor_old_network.get_weights()))

    def sample_experiences(self, batch_size):
        indices = np.random.randint(len(self.queue), size=batch_size)
        batch = np.array([self.queue[index] for index in indices])
        # states, actions, rewards, next_states, dones = [
        #    np.array([experience[field_index] for experience in batch])
        #    for field_index in range(5)]
        return batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3], batch[:, 4]
        # return states, actions, rewards, next_states, dones

    def get_old_prediction(self, s):
        # print(s.shape)
        s = np.reshape(s, (-1, self.input_shape[0]))
        # print(s.shape)
        return self.actor_old_network.predict_on_batch(s)

    def store_experience(self, state, action, reward, state_, done):
        self.queue.append((state, action, reward, state_, done))

#  #      # Inteligente - devolve a ação
  #      #  print(obs)
  #      act = self.epsilon_greedy_policy(obs, self.epsilon)
  #      # act = self.boltzman_exploration_policy(obs)
  #      u = tf.one_hot(act, self.output_shape)
  #      # expected_rewards = self.model.predict(np.reshape(
  #      #    obs, self.input_shape))
  #      # action_do = tf.reduce_sum(expected_rewards * tf.one_hot())
  #      # u = tf.one_hot(self.policy(expected_rewards),
  #      #               depth=self.env.action_space[0].n)
  #
  #      return np.concatenate([u, np.zeros(self.env.world.dim_c)])
  #
  #  def epsilon_greedy_policy(self, state, epsilon=0):
  #      if np.random.rand() < epsilon:
  #          return np.random.randint(self.output_shape)
  #      Q_values = self.model.predict(state[np.newaxis])
  #      return np.argmax(Q_values[0])

#
    #    # Pick random move, in which moves with higher probability are
    #    # more likely to be chosen, but it is obviously not guaranteed
    #    rand_val = random.uniform(0, 1)
    #    prob_sum = 0
    #    for i, prob in enumerate(action_probs):
    #        prob_sum += prob
    #        if rand_val <= prob_sum:
    #            return i
    #    return np.random.randint(self.output_shape)
#

#
    # def play_one_step(self, env, state, epsilon):
    #    action = self.epsilon_greedy_policy(state, epsilon)
    #    next_state, reward, done, info = env.step(action)
    #    self.queue.append((state, action, reward, next_state, done))
    #    return next_state, reward, done, info
#
    # def policy(self, expected_rewards):
    #    return np.argmax(expected_rewards)
#

#
    # def train(self, state=0, action=0, reward=0, state_=0):
    #    # Q(S,A) = Q(S,A) + lr * (R + g * max(Q(state_)) - Q(S,A))
    #    states, actions, rewards, next_states, dones = self.sample_experiences(
    #        self.batch_size)
#
    #    for _ in range(self.epochs):
    #        self.train_count += 1
#
    #        # tf.summary.trace_on(graph=True, profiler=True)
    #        next_Q_values = self.model.predict(next_states)
#
    #        next_Q_values = next_Q_values.squeeze()
#
    #        max_next_Q_values = np.max(next_Q_values, axis=1)
    #        target_Q_values = (rewards +
    #                           (1 - dones) * self.gamma * max_next_Q_values)[np.newaxis]
#
    #        mask = tf.one_hot(actions, self.output_shape)
#
    #        self.model.fit(x=states, y=target_Q_values, verbose=0)
#
    #        # with tf.GradientTape() as tape:
    #        #    all_Q_values = self.model(states)
    #        #    Q_values = tf.reduce_sum(
    #        #        all_Q_values * mask, axis=1, keepdims=False)[np.newaxis]
##
    #        #    # print(Q_values, target_Q_values)
##
    #        #    loss = tf.reduce_mean(self.loss_fn(
    #        #        target_Q_values, Q_values)(target_Q_values, Q_values))
##
    #        #grads = tape.gradient(loss, self.model.trainable_variables)
##
    #        # self.optimizer.apply_gradients(
    #        #    zip(grads, self.model.trainable_variables))
#
    #        self.train_loss(loss)
    #        self.train_accuracy(target_Q_values, Q_values)
#
    #        self.epsilon = self.epsilon * 0.95 if self.epsilon > 0.01 else 0.01
#
    #        with self.train_summary_writer.as_default():
    #            tf.summary.scalar(
    #                'loss', self.train_loss.result(), step=self.train_count)
    #            tf.summary.scalar(
    #                'accuracy', self.train_accuracy.result(), step=self.train_count)
    #            # tf.summary.trace_export(
    #            #    name="train",
    #            #    step=epoch,
    #            #    profiler_outdir=self.run_logdir)
#
    #        # template = 'Epoch {}, Loss: {}, Accuracy: {}, Losses: {}'
    #        # print(template.format(epoch+1,
    #        #                      self.train_loss.result(),
    #        #                      self.train_accuracy.result()*100,
    #        #                      loss))
#
    #        # Reset metrics every epoch
    #        self.train_loss.reset_states()
    #        self.train_accuracy.reset_states()
#
    #        self.queue.clear()
#
    #        # y = []
    #        # x = []
    #        # for step in self.queue:
    #        #    x.append(step[0])
    #        #    pred = self.model.predict(step[0])
    #        #
    #        #    expected_reward = tf.reduce_sum(
    #        #        pred * step[1], axis=1, keepdims=True)
    #        #
    #        #    exp_rewads_ = self.model.predict(step[3])
    #        #
    #        #    reward_groundtruth = expected_reward + self.lr * \
    #        #        (step[2] + self.gamma * np.max(exp_rewads_) - expected_reward)
    #        #
    #        #    y.append(np.reshape(reward_groundtruth *
    #        #                        step[1], self.output_shape))
    #        #
    #        # x = np.reshape(x, (1, -1, 18))
    #        # y = np.reshape(y, (1, -1, 5))
    #        # return self.model.fit(x, y, epochs=self.epochs, verbose=0, batch_size=self.memory)
#
