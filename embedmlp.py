import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import time


class EmbedMLP(tf.keras.Model):
    def __init__(self, n_feats, output_dim, architecture, n_investors, embedding_size, dropout_rates={'input': 0., 'hidden': 0.},
                 weight_decay={'l1': 0., 'l2': 0.}, gamma=2., name='EmbedMLP'):
        super().__init__(name=name)
        self.model_name = name
        self.n_feats = n_feats
        self.output_dim = output_dim
        self.architecture = architecture
        self.n_investors = n_investors
        self.embedding_size = embedding_size
        self.dropout_rates = dropout_rates
        self.weight_decay = weight_decay

        self.gamma = gamma
        self.training = False  # Flag used for correct usage of BatchNorm at serving time.
        self.epsilon = 1e-6
        self.opt = None

        self.embedding = tf.keras.layers.Embedding(input_dim=n_investors, output_dim=embedding_size,
                                                   embeddings_initializer='normal', name='investor_embed')

        self.input_bn = tf.keras.layers.BatchNormalization(name='input_bn')
        self.input_drop = tf.keras.layers.Dropout(rate=dropout_rates['input'], name='input_drop')

        for layer_idx, layer_size in enumerate(self.architecture):
            l = tf.keras.layers.Dense(units=layer_size, activation=None, kernel_initializer='he_normal',
                                      kernel_regularizer=tf.keras.regularizers.l1_l2(**weight_decay),
                                      name=f'l{layer_idx}')
            bn = tf.keras.layers.BatchNormalization(name=f'bn{layer_idx}')
            drop = tf.keras.layers.Dropout(rate=dropout_rates['hidden'], name=f'drop{layer_idx}')
            setattr(self, f'l{layer_idx}', l)
            setattr(self, f'bn{layer_idx}', bn)
            setattr(self, f'drop{layer_idx}', drop)

        self.out = tf.keras.layers.Dense(units=output_dim, activation=None, name='output',
                                         kernel_regularizer=tf.keras.regularizers.l1_l2(**weight_decay))

    @tf.function(experimental_relax_shapes=True)
    def call(self, num_input, cat_input):

        x = self.input_bn(num_input, training=self.training)
        x = self.input_drop(x, training=self.training)

        embed = self.embedding(cat_input)
        embed = tf.reshape(embed, (tf.shape(embed)[0], self.embedding_size))
        x = tf.concat([x, embed], axis=1)

        for layer_idx, _ in enumerate(self.architecture):
            l = getattr(self, f'l{layer_idx}')
            bn = getattr(self, f'bn{layer_idx}')
            drop = getattr(self, f'drop{layer_idx}')
            x = l(x)
            x = bn(x, training=self.training)
            x = tf.nn.relu(x)
            x = drop(x, training=self.training)

        out = self.out(x)
        out = tf.nn.softmax(out)
        return out

    def fake_call(self):

        # To be able to save & load easily, we require a 'build' function that sets all
        # tensors.
        fake_num = np.random.randn(1, self.n_feats).astype(np.float32)
        fake_cat = np.array([0])
        self(fake_num, fake_cat)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, num_input, cat_input, y_true):

        with tf.GradientTape() as tape:
            y_pred = self(num_input, cat_input)
            output_loss = self.focal_loss(y_true, y_pred)
            regularization = sum(self.losses)
            loss = output_loss + regularization

        gradients = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.trainable_variables))
        return float(loss), float(output_loss)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, num_input, cat_input, y_true):

        y_pred = self(num_input, cat_input)
        output_loss = self.focal_loss(y_true, y_pred)
        regularization = sum(self.losses)
        loss = output_loss + regularization
        return float(loss), float(output_loss)

    @tf.function(experimental_relax_shapes=True)
    def focal_loss(self, y_true, y_pred):
        adaptive_weights = tf.math.pow(1 - y_pred + self.epsilon, self.gamma)
        return - tf.reduce_mean(tf.reduce_sum(adaptive_weights * y_true * tf.math.log(y_pred + self.epsilon), axis=1))

    def fit(self, train_data, val_data, n_epochs=10, batch_size=32, optimizer='nadam', learning_rate=1e-3,
            patience=10, lookahead=True, save_path='', seed=0):

        # For reproducibility of results.
        tf.random.set_seed(seed)
        np.random.seed(seed)

        if self.opt is None:
            if optimizer == 'adam':
                self.opt = tf.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer == 'nadam':
                self.opt = tf.optimizers.Nadam(learning_rate=learning_rate)
            elif optimizer == 'rmsprop':
                self.opt = tf.optimizers.RMSprop(learning_rate=learning_rate)
            elif optimizer == 'radam':
                self.opt = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)
            else:
                raise ValueError('Optimizer not recognized. Try in {adam, nadam, radam, rmsprop}.')

            if lookahead:
                self.opt = tfa.optimizers.Lookahead(self.opt, sync_period=5, slow_step_size=0.5)

        else:
            self.opt.learning_rate = learning_rate

        train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        train_dataset = train_dataset.shuffle(buffer_size=8192)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
        val_dataset = val_dataset.batch(val_data[0].shape[0])
        val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        best_val_loss = np.inf
        counter = 0

        recorded_metrics = ['train_loss', 'train_output_loss', 'val_loss', 'val_output_loss']
        history = {k: [] for k in recorded_metrics}

        for epoch in range(n_epochs):

            epoch_duration = time.time()

            # ===== Training =====
            self.training = True

            train_loss = 0.
            train_output_loss = 0.
            train_n_batches = 0
            for step, (x_num, x_cat, y) in enumerate(train_dataset):
                batch_losses = self.train_step(x_num, x_cat, y)
                train_loss += batch_losses[0]
                train_output_loss += batch_losses[1]
                train_n_batches += 1

            train_loss /= train_n_batches
            train_output_loss /= train_n_batches

            # ===== Validation =====
            self.training = False

            val_loss = 0.
            val_output_loss = 0.
            val_n_batches = 0
            for step, (x_num, x_cat, y) in enumerate(val_dataset):
                batch_losses = self.test_step(x_num, x_cat, y)
                val_loss += batch_losses[0]
                val_output_loss += batch_losses[1]
                val_n_batches += 1

            val_loss /= val_n_batches
            val_output_loss /= val_n_batches

            for metric in recorded_metrics:
                history[metric].append(eval(metric).numpy())
            epoch_duration = time.time() - epoch_duration

            print('Epoch {0}/{1} | {2:.2f}s | (train) loss: {3:.5f} - out loss: {4:.5f}'
                  .format(epoch + 1, n_epochs, epoch_duration, train_loss, train_output_loss))
            print('(val) loss: {0:.5f} - out loss: {1:.5f}'.format(val_loss, val_output_loss))

            # ===== Early stopping =====
            if val_loss < best_val_loss:
                print('Best val loss beaten, from {0:.5f} to {1:.5f}. Saving model.\n'
                      .format(best_val_loss, val_loss))
                best_val_loss = val_loss
                counter = 0
                self.save_weights(f'{save_path}{self.model_name}.h5')
            else:
                counter += 1
                if counter == patience:
                    print('{0} epochs performed without improvement. Stopping training.\n'.format(patience))
                    break
                else:
                    print('{0}/{1} epochs performed without improvement. Best val loss: {2:.5f}\n'.format(counter, patience, best_val_loss))

        # Loading best weights.
        self.load_weights(f'{save_path}{self.model_name}.h5')
        history = pd.DataFrame.from_dict(history)
        history['epoch'] = range(len(history))
        return history

    def predict(self, pred_data, batch_size=-1):

        self.training = False
        pred_dataset = tf.data.Dataset.from_tensor_slices(pred_data)

        if batch_size == -1:
            pred_dataset = pred_dataset.batch(pred_data[0].shape[0])
        else:
            pred_dataset = pred_dataset.batch(batch_size)
            pred_dataset = pred_dataset.prefetch(buffer_size=batch_size)

        for step, (x_num, x_cat) in enumerate(pred_dataset):

            pred = self(x_num, x_cat)
            if step == 0:
                preds = pred.numpy()
            else:
                preds = np.concatenate((preds, pred.numpy()))

        return preds
