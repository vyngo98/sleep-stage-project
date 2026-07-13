import keras
import tensorflow as tf

def lstm_feature_model(dim_features=132, n_classes=4):
    # Input layer: 132 features
    input_layer = tf.keras.layers.Input(shape=(dim_features, 1))

    # First perceptron layer (Dense)
    x = tf.keras.layers.Dense(32, activation='sigmoid')(input_layer)

    # LSTM layers (Bidirectional with 32 forward + 32 backward = 64 total)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=True, activation='sigmoid'))(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=True, activation='sigmoid'))(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, activation='sigmoid'))(x)

    # Dense layer with 32 units
    x = tf.keras.layers.Dense(32, activation='sigmoid')(x)

    # Output layer with 4 classes
    output_layer = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    # Build model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    return model

# model = lstm_feature_model(dim_features=118, n_classes=4)
# model.summary()

def gamma_cross_entropy_loss(y_true, y_pred, gamma=0.3):
    """
    Custom loss: L = sum_i [ y_i * (-log(f_i(x)))^γ + (1 - y_i) * (-log(1 - f_i(x)))^γ ]
    Args:
        y_true: ground-truth labels, shape (batch, num_classes)
        y_pred: predicted probabilities (after sigmoid or softmax), same shape
        gamma: exponent parameter (controls gradient)
    """
    eps = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
    loss = -tf.reduce_sum(y_true * ((1 - y_pred) ** gamma) * tf.math.log(y_pred), axis=-1)
    return tf.reduce_mean(loss)


import tensorflow as tf
from tensorflow.keras import layers, Model
import math


class PreconvConv1D(layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=5, growth_rate=4, **kwargs):
        super().__init__(**kwargs)
        inter_channels = growth_rate * out_channels  # channel-growth controlling layer
        self.pre_conv = tf.keras.Sequential([
            layers.Conv1D(inter_channels, kernel_size=1, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.conv = layers.Conv1D(out_channels, kernel_size=kernel_size, padding="same", use_bias=False)
        self.bn = layers.BatchNormalization()
        self.act = layers.ReLU()

    def call(self, x, training=False):
        x = self.pre_conv(x, training=training)
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.act(x)
        return x

class DenseBlock(layers.Layer):
    def __init__(self, num_layers=2, growth_rate=4, out_channels=32, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.blocks = []
        for _ in range(num_layers):
            self.blocks.append(
                PreconvConv1D(in_channels=None,  # dynamic channels handled in call
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              growth_rate=growth_rate)
            )

    def call(self, x, training=False):
        features = [x]
        for block in self.blocks:
            out = block(tf.concat(features, axis=-1), training=training)
            features.append(out)
        return tf.concat(features, axis=-1)


class TransitionBlock(layers.Layer):
    def __init__(self, reduction=0.5, pool_size=2, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.pool = layers.AveragePooling1D(pool_size=pool_size, strides=pool_size, padding="same")

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        out_channels = int(in_channels * self.reduction)
        self.conv = layers.Conv1D(out_channels, kernel_size=1, padding="same", use_bias=False)
        self.bn = layers.BatchNormalization()
        self.act = layers.ReLU()

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.act(x)
        x = self.pool(x)
        return x

def build_ppg_densenet(input_length=1920, num_classes=4):
    inputs = layers.Input(shape=(input_length, 1))

    # Low convolution block
    x = layers.Conv1D(32, kernel_size=21, strides=5, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(32, kernel_size=21, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling1D(pool_size=2)(x)  # Output ≈ (192, 32)

    # Dense + Transition blocks
    for i in range(4):
        x = DenseBlock(num_layers=2, out_channels=32, growth_rate=4)(x)
        if i < 3:  # skip transition after last dense block
            x = TransitionBlock(reduction=0.5, pool_size=2)(x)

    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Classifier
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="PPG_DenseNet")


weights = tf.constant([2.07, 0.88, 0.72], dtype=tf.float32)


def weighted_categorical_crossentropy(y_true, y_pred):
    class_idx = tf.argmax(y_true, axis=-1)
    sample_weights = tf.gather(weights, class_idx)

    loss = tf.keras.losses.categorical_crossentropy(
        y_true, y_pred, from_logits=False
    )

    return loss * sample_weights

class SleepStageModel:
    def __init__(self, n_epochs, len_epoch, output_len):
        super(SleepStageModel, self).__init__()
        self.n_epochs = n_epochs
        self.len_epoch = len_epoch
        self.output_len = output_len

    def make(self, cons=tf.keras.regularizers.L1(0.25)):
        ihr_inputs = layers.Input(shape=(self.n_epochs, self.len_epoch, 1), name='ecg_input')
        x = layers.Conv1D(filters=8, kernel_size=1, padding='same', strides=1,
                            kernel_regularizer=cons, name='conv1')(ihr_inputs)

        res = x
        x = layers.Conv1D(filters=64, kernel_size=3, padding='same', strides=1,
                            kernel_regularizer=cons, dilation_rate=1, name='conv2')(x)
        x = layers.LeakyReLU(alpha=0.15)(x)
        x = layers.Conv1D(filters=64, kernel_size=3, padding='same', strides=1,
                            kernel_regularizer=cons, dilation_rate=1, name='conv3')(x)
        x = layers.LeakyReLU(alpha=0.15)(x)
        x = layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2))(x)
        res = layers.Conv1D(filters=64, kernel_size=1, padding='same', strides=1,
                              kernel_regularizer=cons)(res)
        res = layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2))(res)
        x = layers.Add()([x, res])

        res = x
        x = layers.Conv1D(filters=64, kernel_size=3, padding='same', strides=1,
                            kernel_regularizer=cons, dilation_rate=1, name='conv4')(x)
        x = layers.LeakyReLU(alpha=0.15)(x)
        x = layers.Conv1D(filters=64, kernel_size=3, padding='same', strides=1,
                            kernel_regularizer=cons, dilation_rate=1, name='conv5')(x)
        x = layers.LeakyReLU(alpha=0.15)(x)
        x = layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2))(x)

        res = layers.Conv1D(filters=64, kernel_size=1, padding='same', strides=1,
                              kernel_regularizer=cons)(res)
        res = layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2))(res)
        x = layers.Add()([x, res])

        res = x
        x = layers.Conv1D(filters=64, kernel_size=3, padding='same', strides=1,
                            kernel_regularizer=cons, dilation_rate=1, name='conv6')(x)
        x = layers.LeakyReLU(alpha=0.15)(x)
        x = layers.Conv1D(filters=64, kernel_size=3, padding='same', strides=1,
                            kernel_regularizer=cons, dilation_rate=1, name='conv7')(x)
        x = layers.LeakyReLU(alpha=0.15)(x)
        x = layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2))(x)

        res = layers.Conv1D(filters=64, kernel_size=1, padding='same', strides=1, kernel_regularizer=cons)(res)
        res = layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2))(res)
        x = layers.Add()([x, res])

        x = layers.Flatten()(x)
        x = layers.Reshape((self.n_epochs, 2048))(x)
        x = layers.Dense(128, activation='relu')(x)
        x = tf.expand_dims(x, axis=1)
        res = x

        x = layers.Conv1D(filters=128, kernel_size=7, dilation_rate=2, padding='same',
                            kernel_regularizer=cons, name='conv8')(x)
        x = layers.LeakyReLU(alpha=0.15)(x)
        x = layers.Conv1D(filters=128, kernel_size=7, dilation_rate=4, padding='same',
                            kernel_regularizer=cons, name='conv9')(x)
        x = layers.LeakyReLU(alpha=0.15)(x)
        x = layers.Conv1D(filters=128, kernel_size=7, dilation_rate=8, padding='same',
                            kernel_regularizer=cons, name='conv10')(x)
        x = layers.LeakyReLU(alpha=0.15)(x)
        x = layers.Conv1D(filters=128, kernel_size=7, dilation_rate=16, padding='same',
                            kernel_regularizer=cons, name='conv11')(x)
        x = layers.LeakyReLU(alpha=0.15)(x)
        x = layers.Conv1D(filters=128, kernel_size=7, dilation_rate=32, padding='same',
                            kernel_regularizer=cons, name='conv12')(x)
        x = layers.LeakyReLU(alpha=0.15)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Add()([x, res])

        res = x
        x = layers.Conv1D(filters=128, kernel_size=7, dilation_rate=2, padding='same',
                            kernel_regularizer=cons, name='conv13')(x)
        x = layers.LeakyReLU(alpha=0.15)(x)
        x = layers.Conv1D(filters=128, kernel_size=7, dilation_rate=4, padding='same',
                            kernel_regularizer=cons, name='conv14')(x)
        x = layers.LeakyReLU(alpha=0.15)(x)
        x = layers.Conv1D(filters=128, kernel_size=7, dilation_rate=8, padding='same',
                            kernel_regularizer=cons, name='conv15')(x)
        x = layers.LeakyReLU(alpha=0.15)(x)
        x = layers.Conv1D(filters=128, kernel_size=7, dilation_rate=16, padding='same',
                            kernel_regularizer=cons, name='conv16')(x)
        x = layers.LeakyReLU(alpha=0.15)(x)
        x = layers.Conv1D(filters=128, kernel_size=7, dilation_rate=32, padding='same',
                            kernel_regularizer=cons, name='conv17')(x)
        x = layers.LeakyReLU(alpha=0.15)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Add()([x, res])
        x = layers.Conv1D(filters=2, kernel_size=1, padding='same', strides=1, dilation_rate=1,
                            activation='softmax', kernel_regularizer=cons, name='conv18')(x)
        outputs = tf.squeeze(x, axis=1)
        model = tf.keras.Model(inputs=ihr_inputs, outputs=outputs)
        return model

class ExpLayer(layers.Layer):
    def call(self, inputs):
        # inputs here is a Keras tensor, which can be directly passed to tf.exp
        return tf.expand_dims(inputs, axis=1)

class SqueezeLayer(layers.Layer):
    def call(self, inputs):
        # inputs here is a Keras tensor, which can be directly passed to tf.exp
        return tf.squeeze(inputs, axis=1)

# class SleepStageIHRnACCModel:
#     def __init__(self, n_epochs, len_epoch_ihr, len_epoch_acc, output_len):
#         super().__init__()
#         self.n_epochs = n_epochs
#         self.len_epoch_ihr = len_epoch_ihr   # 256
#         self.len_epoch_acc = len_epoch_acc   # 128
#         self.output_len = output_len
#
#     def epoch_encoder(self, x, cons, name_prefix):
#         """Shared encoder structure for one modality"""
#         x = layers.Conv2D(8, 1, padding='same',
#                           kernel_regularizer=cons,
#                           name=f'{name_prefix}_conv1')(x)
#
#         for i in range(3):
#             res = x
#             x = layers.Conv2D(64, 3, padding='same',
#                               kernel_regularizer=cons,
#                               name=f'{name_prefix}_conv_{i}_1')(x)
#             x = layers.LeakyReLU(0.15)(x)
#             x = layers.Conv2D(64, 3, padding='same',
#                               kernel_regularizer=cons,
#                               name=f'{name_prefix}_conv_{i}_2')(x)
#             x = layers.LeakyReLU(0.15)(x)
#
#             x = layers.MaxPool2D(pool_size=(1, 2))(x)
#             res = layers.Conv2D(64, 1, padding='same',
#                                 kernel_regularizer=cons)(res)
#             res = layers.MaxPool2D(pool_size=(1, 2))(res)
#             x = layers.Add()([x, res])
#
#         x = layers.Flatten()(x)
#         x = layers.Reshape((self.n_epochs, -1))(x)
#         x = layers.Dense(128, activation='relu')(x)
#         return x
#
#
#
#     def make(self, cons=tf.keras.regularizers.L1(0.25)):
#
#         # ---------------- Inputs ----------------
#         ihr_inputs = layers.Input(
#             shape=(self.n_epochs, self.len_epoch_ihr, 1),
#             name='ihr_input'
#         )
#
#         acc_inputs = layers.Input(
#             shape=(self.n_epochs, self.len_epoch_acc, 1),
#             name='acc_input'
#         )
#
#         # ---------------- Encoders ----------------
#         ihr_feat = self.epoch_encoder(ihr_inputs, cons, 'ihr')
#         acc_feat = self.epoch_encoder(acc_inputs, cons, 'acc')
#
#         # ---------------- Fusion ----------------
#         x = layers.Concatenate(axis=-1)([ihr_feat, acc_feat])
#         x = layers.Dense(256, activation='relu')(x)
#
#         # ---------------- Temporal Modeling ----------------
#         # x = tf.expand_dims(x, axis=1)
#         # x = ExpLayer()(x)
#         res = x
#
#         for d, k in zip([2, 4, 8, 16, 32], range(8, 13)):
#             x = layers.Conv1D(
#                 256, 7, dilation_rate=d, padding='same',
#                 kernel_regularizer=cons,
#                 name=f'conv{k}'
#             )(x)
#             x = layers.LeakyReLU(0.15)(x)
#
#         x = layers.Dropout(0.2)(x)
#         x = layers.Add()([x, res])
#
#         # ---------------- Output ----------------
#         outputs = layers.Conv1D(
#             self.output_len, 1, padding='same',
#             activation='softmax',
#             kernel_regularizer=cons,
#             name='classifier'
#         )(x)
#
#         # outputs = tf.squeeze(x, axis=1)
#         # outputs = SqueezeLayer()(x)
#
#         model = tf.keras.Model(
#             inputs=[ihr_inputs, acc_inputs],
#             outputs=outputs
#         )
#
#         return model

class SleepStageIHRnACCModel:
    def __init__(self, n_epochs, len_epoch_ihr, len_epoch_acc, output_len):
        self.n_epochs = n_epochs          # 299
        self.len_epoch_ihr = len_epoch_ihr  # 256
        self.len_epoch_acc = len_epoch_acc  # 128
        self.output_len = output_len      # n_classes

    # --------------------------------------------------
    # Epoch-level encoder (Conv1D per epoch)
    # --------------------------------------------------
    def epoch_encoder(self, x, cons, name):
        """
        Input:  (batch, epochs, samples, 1)
        Output: (batch, epochs, 128)
        """

        x = layers.TimeDistributed(
            layers.Conv1D(8, 1, padding='same',
                          kernel_regularizer=cons),
            name=f'{name}_conv0'
        )(x)

        for i in range(3):
            res = x

            # x = layers.TimeDistributed(
            #     layers.Conv1D(32, 3, padding='same',
            #                   kernel_regularizer=cons),
            #     name=f'{name}_conv{i}_1'
            # )(x)
            # x = layers.LeakyReLU(0.15)(x)

            x = layers.TimeDistributed(
                layers.Conv1D(32, 3, padding='same',
                              kernel_regularizer=cons),
                name=f'{name}_conv{i}_2'
            )(x)
            x = layers.LeakyReLU(0.15)(x)

            x = layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2))(x)

            res = layers.TimeDistributed(
                layers.Conv1D(32, 1, padding='same',
                              kernel_regularizer=cons)
            )(res)

            # res = layers.TimeDistributed(
            #     layers.MaxPooling1D(2)
            # )(res)

            res = layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2))(res)

            x = layers.Add()([x, res])

        x = layers.TimeDistributed(
            layers.Flatten(),
            name=f'{name}_flatten'
        )(x)

        x = layers.TimeDistributed(
            layers.Dense(128, activation='relu'),
            name=f'{name}_dense'
        )(x)

        return x

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------
    def make(self, cons=tf.keras.regularizers.L1(0.25)):

        # ---------------- Inputs ----------------
        ihr_inputs = layers.Input(
            shape=(self.n_epochs, self.len_epoch_ihr, 1),
            name='ihr_input'
        )

        acc_inputs = layers.Input(
            shape=(self.n_epochs, self.len_epoch_acc, 1),
            name='acc_input'
        )

        # ---------------- Epoch encoders ----------------
        ihr_feat = self.epoch_encoder(ihr_inputs, cons, 'ihr')
        acc_feat = self.epoch_encoder(acc_inputs, cons, 'acc')

        # ---------------- Fusion ----------------
        x = layers.Concatenate(axis=-1, name='fusion')([ihr_feat, acc_feat])
        x = layers.Dense(256, activation='relu', name='fusion_dense')(x)

        # ---------------- Temporal modeling (TCN) ----------------
        res = x

        # for d in [1, 2, 4, 8, 16]:
        for d in [2, 4, 8]:
            x = layers.Conv1D(
                filters=256,
                kernel_size=7,
                dilation_rate=d,
                padding='same',
                kernel_regularizer=cons
            )(x)
            x = layers.LeakyReLU(0.15)(x)

        x = layers.Dropout(0.2)(x)
        x = layers.Add()([x, res])

        # ---------------- Output ----------------
        outputs = layers.Conv1D(
            self.output_len,
            kernel_size=1,
            activation='softmax',
            padding='same',
            kernel_regularizer=cons,
            name='classifier'
        )(x)

        return tf.keras.Model(
            inputs=[ihr_inputs, acc_inputs],
            outputs=outputs,
            name='SleepStage_IHR_ACC'
        )


class CoAttention(tf.keras.Layer):
    def call(self, inputs):
        Q, D = inputs     # shapes: Q=(B, n+1, l), D=(B, m+1, l)

        # 1. D_t = D.transpose(1, 2)
        D_t = tf.transpose(D, perm=[0, 2, 1])   # (B, l, m+1)

        # 2. L = Q @ D_t  (torch.bmm)
        L = tf.matmul(Q, D_t)                  # (B, n+1, m+1)

        # 3. A_Q_ = softmax(L, dim=1)
        A_Q_ = tf.nn.softmax(L, axis=1)        # (B, n+1, m+1)

        # 4. A_Q = A_Q_.transpose(1,2)
        A_Q = tf.transpose(A_Q_, perm=[0, 2, 1])  # (B, m+1, n+1)

        # 5. C_Q = D_t @ A_Q
        C_Q = tf.matmul(D_t, A_Q)              # (B, l, n+1)

        # 6. Q_t = Q.transpose(1, 2)
        Q_t = tf.transpose(Q, perm=[0, 2, 1])  # (B, l, n+1)

        # 7. A_D = softmax(L, dim=2)
        A_D = tf.nn.softmax(L, axis=2)         # (B, n+1, m+1)

        # 8. concatenate (Q_t, C_Q)
        QC_cat = tf.concat([Q_t, C_Q], axis=1) # (B, 2l, n+1)

        # 9. C_D = QC_cat @ A_D
        C_D = tf.matmul(QC_cat, A_D)           # (B, 2l, m+1)

        # 10. C_D_t = C_D.transpose(1,2)
        C_D_t = tf.transpose(C_D, perm=[0, 2, 1])  # (B, m+1, 2l)

        concat_tensor = tf.keras.layers.Concatenate(axis=-1)([C_D_t, D])
        return concat_tensor

class SleepStageSmallIHR_ACC:
    def __init__(self,
                 n_epochs=250,
                 len_ihr=256,
                 len_acc=128,
                 n_classes=3):
        self.n_epochs = n_epochs
        self.len_ihr = len_ihr
        self.len_acc = len_acc
        self.n_classes = n_classes

    # --------------------------------------------------
    # Small per-epoch encoder
    # --------------------------------------------------
    def epoch_encoder(self, x, name):
        """
        Input:  (B, epochs, samples, 1)
        Output: (B, epochs, 64)
        """

        x = layers.TimeDistributed(
            layers.Conv1D(16, 5, padding='same'),
            name=f'{name}_conv1'
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.ReLU())(x)
        x = layers.TimeDistributed(layers.MaxPooling1D(2))(x)

        x = layers.TimeDistributed(
            layers.Conv1D(32, 3, padding='same'),
            name=f'{name}_conv2'
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.ReLU())(x)
        x = layers.TimeDistributed(layers.MaxPooling1D(2))(x)

        x = layers.TimeDistributed(
            layers.GlobalAveragePooling1D(),
            name=f'{name}_gap'
        )(x)

        x = layers.TimeDistributed(
            layers.Dense(64, activation='relu'),
            name=f'{name}_dense'
        )(x)

        return x

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------
    def make(self):

        # ---------------- Inputs ----------------
        ihr_input = layers.Input(
            shape=(self.n_epochs, self.len_ihr, 1),
            name='ihr_input'
        )

        acc_input = layers.Input(
            shape=(self.n_epochs, self.len_acc, 1),
            name='acc_input'
        )

        # ---------------- Epoch encoders ----------------
        ihr_feat = self.epoch_encoder(ihr_input, 'ihr')
        acc_feat = self.epoch_encoder(acc_input, 'acc')

        # ---------------- Fusion ----------------
        x = layers.Concatenate(name='fusion')([ihr_feat, acc_feat])
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        # ---------------- Temporal modeling ----------------
        x = layers.Bidirectional(
            layers.GRU(
                64,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.1
            ),
            name='bigru'
        )(x)

        # ---------------- Output ----------------
        outputs = layers.TimeDistributed(
            layers.Dense(self.n_classes, activation='softmax'),
            name='classifier'
        )(x)

        return tf.keras.Model(
            inputs=[ihr_input, acc_input],
            outputs=outputs,
            name='SleepStage_Small_IHR_ACC'
        )

class SleepStageSmallIHR_ACC_CoAttention:
    def __init__(self,
                 n_epochs=250,
                 len_ihr=256,
                 len_acc=128,
                 n_classes=3):
        self.n_epochs = n_epochs
        self.len_ihr = len_ihr
        self.len_acc = len_acc
        self.n_classes = n_classes

    # --------------------------------------------------
    # Small per-epoch encoder
    # --------------------------------------------------
    def epoch_encoder(self, x, name):
        """
        Input:  (B, epochs, samples, 1)
        Output: (B, epochs, 64)
        """

        # x = layers.TimeDistributed(
        #     layers.Conv1D(8, 5, padding='same', kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),
        #     name=f'{name}_conv1'
        # )(x)
        x = layers.TimeDistributed(
            layers.Conv1D(8, 5, padding='same'),
            name=f'{name}_conv1'
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.ReLU())(x)
        x = layers.TimeDistributed(layers.MaxPooling1D(2))(x)

        x = layers.TimeDistributed(
            layers.Conv1D(16, 3, padding='same'),
            name=f'{name}_conv2'
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.ReLU())(x)
        x = layers.TimeDistributed(layers.MaxPooling1D(2))(x)

        x = layers.TimeDistributed(
            layers.GlobalAveragePooling1D(),
            name=f'{name}_gap'
        )(x)

        # x = layers.TimeDistributed(
        #     layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),
        #     name=f'{name}_dense'
        # )(x)

        x = layers.TimeDistributed(
            layers.Dense(32, activation='relu'),
            name=f'{name}_dense'
        )(x)

        return x

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------
    def make(self):

        # ---------------- Inputs ----------------
        ihr_input = layers.Input(
            shape=(self.n_epochs, self.len_ihr, 1),
            name='ihr_input'
        )

        acc_input = layers.Input(
            shape=(self.n_epochs, self.len_acc, 1),
            name='acc_input'
        )

        # ---------------- Epoch encoders ----------------
        ihr_feat = self.epoch_encoder(ihr_input, 'ihr')
        acc_feat = self.epoch_encoder(acc_input, 'acc')

        # ----------------- CoAttention ------------------
        C_D_12 = CoAttention()([ihr_feat, acc_feat])
        C_D_21 = CoAttention()([acc_feat, ihr_feat])

        # ---------------- Fusion ----------------
        x = layers.Concatenate(axis=-1)([C_D_12, C_D_21])
        # x = layers.Concatenate(name='fusion')([ihr_feat, acc_feat])

        # x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        # ---------------- Temporal modeling ----------------
        x = layers.Bidirectional(
            layers.GRU(
                32,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.1
            ),
            name='bigru')(x)

        # ---------------- Output ----------------
        outputs = layers.TimeDistributed(
            layers.Dense(self.n_classes, activation='softmax'),
            name='classifier'
        )(x)

        return tf.keras.Model(
            inputs=[ihr_input, acc_input],
            outputs=outputs,
            name='SleepStage_Small_IHR_ACC'
        )

class TemporalAttention(layers.Layer):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=heads, key_dim=dim)
        self.norm = layers.LayerNormalization()
        self.ffn = tf.keras.Sequential([
            layers.Dense(dim * 2, activation='relu'),
            layers.Dense(dim)
        ])

    def call(self, x):
        attn = self.mha(x, x)
        x = self.norm(x + attn)   # residual

        ffn_out = self.ffn(x)
        x = self.norm(x + ffn_out)

        return x

class PositionalEncoding(layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def call(self, x):
        pos = tf.range(tf.shape(x)[1])[:, tf.newaxis]
        i = tf.range(self.d_model)[tf.newaxis, :]
        i = tf.cast(i, 'float32')
        angle_rates = 1 / tf.pow(10000., (2 * (i//2)) / tf.cast(self.d_model, tf.float32))
        angle_rads = tf.cast(pos, tf.float32) * angle_rates

        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        return x + pos_encoding

class NnSoftmax(layers.Layer):
    def call(self, x, axis=-1):
        x = tf.nn.softmax(x, axis=axis)
        return x

class Split(layers.Layer):
    def call(self, x, n=2, axis=-1):
        x = tf.split(x, n, axis=-1)
        return x

class Reshape(layers.Layer):
    def call(self, x, shape):
        x = tf.reshape(x, shape)
        return x

class ImprovedSleepStageSmallIHR_ACC_CoAttention:
    def __init__(self,
                 n_epochs=250,
                 len_ihr=256,
                 len_acc=128,
                 n_classes=3):
        self.n_epochs = n_epochs
        self.len_ihr = len_ihr
        self.len_acc = len_acc
        self.n_classes = n_classes

    @staticmethod
    def cross_attention_3(A, B, C):

        # A attends to (B + C)
        BC = layers.Concatenate(axis=1)([B, C])
        A_out = CoAttention()([A, BC])

        # B attends to (A + C)
        AC = layers.Concatenate(axis=1)([A, C])
        B_out = CoAttention()([B, AC])

        # C attends to (A + B)
        AB = layers.Concatenate(axis=1)([A, B])
        C_out = CoAttention()([C, AB])

        return A_out, B_out, C_out

    def epoch_encoder(self, x, name):
        """
        Input:  (B, epochs, samples, 1)
        Output: (B, epochs, 64)
        """
        kernel_size1 = 7
        kernel_size2 = 5

        # if name == "ihr":
        #     kernel_size1 = 12
        #     kernel_size2 = 10
        # else:
        #     kernel_size1 = 6
        #     kernel_size2 = 5

        x = layers.TimeDistributed(
            layers.Conv1D(8, kernel_size1, padding='same'),
            name=f'{name}_conv1'
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.ReLU())(x)
        x = layers.TimeDistributed(layers.MaxPooling1D(2))(x)

        # if name == "ihr":
        #     x = layers.TimeDistributed(layers.MaxPooling1D(2))(x)

        # x = layers.TimeDistributed(
        #     layers.Conv1D(16, 3, padding='same'),
        #     name=f'{name}_conv2'
        # )(x)

        x = layers.TimeDistributed(
            layers.Conv1D(16, kernel_size2, padding='same'),
            name=f'{name}_conv2'
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.ReLU())(x)
        x = layers.TimeDistributed(layers.MaxPooling1D(2))(x)

        x = layers.TimeDistributed(
            layers.GlobalAveragePooling1D(),
            name=f'{name}_gap'
        )(x)

        x = layers.TimeDistributed(
            layers.Dense(32, activation='relu'),
            name=f'{name}_dense'
        )(x)

        return x

    # @staticmethod
    # def feature_gating(ihr_feat, acc_feat):
    #     concat = layers.Concatenate()([ihr_feat, acc_feat])
    #
    #     gate = layers.Dense(96, activation='sigmoid')(concat)
    #     fused = gate * ihr_feat + (1 - gate) * acc_feat
    #
    #     # gate = layers.Dense(2)(concat)
    #     # gate = NnSoftmax()(gate, axis=-1)
    #     # g1, g2 = Split()(gate, n=2, axis=-1)
    #     # fused = g1 * ihr_feat + g2 * acc_feat
    #
    #     return fused

    @staticmethod
    def feature_gating3(ihr_feat, acc_feat, d_feat):
        concat = layers.Concatenate()([ihr_feat, acc_feat, d_feat])

        # Generate 3 attention weights
        gate_logits = layers.Dense(3)(concat)  # (B, T, 3)
        gate = NnSoftmax()(gate_logits, axis=-1)

        # Split weights
        g_ihr = gate[..., 0:1]
        g_acc = gate[..., 1:2]
        g_d = gate[..., 2:3]

        # Weighted sum
        fused = g_ihr * ihr_feat + g_acc * acc_feat + g_d * d_feat
        fused = layers.MaxPooling1D(2)(fused)
        return fused

    @staticmethod
    def feature_gating(ihr_feat, acc_feat):
        concat = layers.Concatenate()([ihr_feat, acc_feat])

        gate_logits = layers.Dense(2)(concat)
        gate = NnSoftmax()(gate_logits, axis=-1)

        g_ihr = gate[..., 0:1]
        g_acc = gate[..., 1:2]

        fused = g_ihr * ihr_feat + g_acc * acc_feat
        return fused

    def make(self):
        ihr_input = layers.Input(
            shape=(self.n_epochs, self.len_ihr, 1),
            name='ihr_input'
        )

        acc_input = layers.Input(
            shape=(self.n_epochs, self.len_acc, 1),
            name='acc_input'
        )

        # d_input = layers.Input(
        #     shape=(self.n_epochs, self.len_d, 1),
        #     name='d_input'
        # )

        # ---------------- CNN ----------------
        ihr_feat = self.epoch_encoder(ihr_input, 'ihr')
        acc_feat = self.epoch_encoder(acc_input, 'acc')
        # d_feat = self.epoch_encoder(d_input, 'd')

        # ---------------- Cross Attention ----------------
        C_D_12 = CoAttention()([ihr_feat, acc_feat])
        C_D_21 = CoAttention()([acc_feat, ihr_feat])
        # A_out, B_out, C_out = self.cross_attention_3(ihr_feat, acc_feat, d_feat)

        # ---------------- Fusion ----------------
        # fused = layers.Concatenate(axis=-1)([C_D_12, C_D_21])
        fused = self.feature_gating(C_D_12, C_D_21)
        # fused = self.feature_gating3(A_out, B_out, C_out)

        x = layers.Dense(64, activation='relu')(fused)

        # ---------------- Local temporal ----------------
        x = layers.Bidirectional(
            layers.GRU(64, return_sequences=True)
        )(x)

        # Conv1D smoothing
        x = layers.Conv1D(64, kernel_size=9, padding='same', activation='relu')(x)

        # 🔥 ---------------- GLOBAL TEMPORAL ATTENTION ----------------
        x = PositionalEncoding(64)(x)
        # x = TemporalAttention(128)(x)
        x = TemporalAttention(64)(x)

        # ---------------- Residual refinement ----------------
        # x_res = layers.Dense(128)(fused)
        x_res = layers.Dense(64)(fused)
        x = layers.Add()([x, x_res])

        x = layers.LayerNormalization()(x)

        # ---------------- Output ----------------
        outputs = layers.TimeDistributed(
            layers.Dense(self.n_classes, activation='softmax')
        )(x)

        return tf.keras.Model(
            inputs=[ihr_input, acc_input],
            outputs=outputs,
            name='SleepStage_Improved'
        )

class ImprovedSleepStageSmallIHR_ACC_CoAttention2:
    def __init__(self,
                 n_epochs=250,
                 len_ihr=256,
                 len_acc=128,
                 n_classes=3):
        self.n_epochs = n_epochs
        self.len_ihr = len_ihr
        self.len_acc = len_acc
        self.n_classes = n_classes

    @staticmethod
    def cross_attention_3(A, B, C):

        # A attends to (B + C)
        BC = layers.Concatenate(axis=1)([B, C])
        A_out = CoAttention()([A, BC])

        # B attends to (A + C)
        AC = layers.Concatenate(axis=1)([A, C])
        B_out = CoAttention()([B, AC])

        # C attends to (A + B)
        AB = layers.Concatenate(axis=1)([A, B])
        C_out = CoAttention()([C, AB])

        return A_out, B_out, C_out

    def epoch_encoder(self, x, name):
        """
        Input:  (B, epochs, samples, 1)
        Output: (B, epochs, 64)
        """
        kernel_size1 = 7
        kernel_size2 = 5

        # if name == "ihr":
        #     kernel_size1 = 12
        #     kernel_size2 = 10
        # else:
        #     kernel_size1 = 6
        #     kernel_size2 = 5

        x = layers.Conv2D(8, kernel_size1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2, data_format='channels_first')(x)

        # if name == "ihr":
        #     x = layers.TimeDistributed(layers.MaxPooling1D(2))(x)

        # x = layers.TimeDistributed(
        #     layers.Conv1D(16, 3, padding='same'),
        #     name=f'{name}_conv2'
        # )(x)

        x = layers.Conv2D(16, kernel_size2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2, data_format='channels_first')(x)

        # x = layers.GlobalAveragePooling2D(data_format="channels_first")(x)

        x = layers.Dense(32, activation='relu')(x)

        return x


    @staticmethod
    def feature_gating(ihr_feat, acc_feat):
        concat = layers.Concatenate()([ihr_feat, acc_feat])

        gate_logits = layers.Dense(2)(concat)
        gate = NnSoftmax()(gate_logits, axis=-1)

        g_ihr = gate[..., 0:1]
        g_acc = gate[..., 1:2]

        fused = g_ihr * ihr_feat + g_acc * acc_feat
        return fused

    def make(self):
        ihr_input = layers.Input(
            shape=(self.n_epochs, self.len_ihr, 1),
            name='ihr_input'
        )

        acc_input = layers.Input(
            shape=(self.n_epochs, self.len_acc, 1),
            name='acc_input'
        )

        inputs = ConcatLayer()(ihr_input, acc_input, axis=-1)

        # ---------------- CNN ----------------
        x = self.epoch_encoder(inputs, 'ihr_acc')

        # ---------------- Cross Attention ----------------
        # C_D_12 = CoAttention()([ihr_feat, acc_feat])
        # C_D_21 = CoAttention()([acc_feat, ihr_feat])
        # A_out, B_out, C_out = self.cross_attention_3(ihr_feat, acc_feat, d_feat)

        # ---------------- Fusion ----------------
        # fused = layers.Concatenate(axis=-1)([C_D_12, C_D_21])
        # fused = self.feature_gating(C_D_12, C_D_21)
        # fused = self.feature_gating3(A_out, B_out, C_out)

        x_ = layers.Dense(64, activation='relu')(x)

        # ---------------- Local temporal ----------------
        x = layers.Bidirectional(
            layers.GRU(64, return_sequences=True)
        )(x_)

        # Conv1D smoothing
        x = layers.Conv1D(64, kernel_size=9, padding='same', activation='relu')(x)

        # 🔥 ---------------- GLOBAL TEMPORAL ATTENTION ----------------
        x = PositionalEncoding(64)(x)
        # x = TemporalAttention(128)(x)
        x = TemporalAttention(64)(x)

        # ---------------- Residual refinement ----------------
        # x_res = layers.Dense(128)(fused)
        x_res = layers.Dense(64)(x_)
        x = layers.Add()([x, x_res])

        x = layers.LayerNormalization()(x)

        # ---------------- Output ----------------
        outputs = layers.TimeDistributed(
            layers.Dense(self.n_classes, activation='softmax')
        )(x)

        return tf.keras.Model(
            inputs=[ihr_input, acc_input],
            outputs=outputs,
            name='SleepStage_Improved'
        )


class SleepStageSmallIHR_ACC_CoAttention_CRF:
    def __init__(self,
                 n_epochs=250,
                 len_ihr=256,
                 len_acc=128,
                 n_classes=3):
        self.n_epochs = n_epochs
        self.len_ihr = len_ihr
        self.len_acc = len_acc
        self.n_classes = n_classes

    # --------------------------------------------------
    # Small per-epoch encoder
    # --------------------------------------------------
    def epoch_encoder(self, x, name):
        """
        Input:  (B, epochs, samples, 1)
        Output: (B, epochs, 64)
        """

        # x = layers.TimeDistributed(
        #     layers.Conv1D(8, 5, padding='same', kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),
        #     name=f'{name}_conv1'
        # )(x)
        x = layers.TimeDistributed(
            layers.Conv1D(8, 5, padding='same'),
            name=f'{name}_conv1'
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.ReLU())(x)
        x = layers.TimeDistributed(layers.MaxPooling1D(2))(x)

        x = layers.TimeDistributed(
            layers.Conv1D(16, 3, padding='same'),
            name=f'{name}_conv2'
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.ReLU())(x)
        x = layers.TimeDistributed(layers.MaxPooling1D(2))(x)

        x = layers.TimeDistributed(
            layers.GlobalAveragePooling1D(),
            name=f'{name}_gap'
        )(x)

        # x = layers.TimeDistributed(
        #     layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),
        #     name=f'{name}_dense'
        # )(x)

        x = layers.TimeDistributed(
            layers.Dense(32, activation='relu'),
            name=f'{name}_dense'
        )(x)

        return x

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------
    def make(self):

        # ---------------- Inputs ----------------
        ihr_input = layers.Input(
            shape=(self.n_epochs, self.len_ihr, 1),
            name='ihr_input'
        )

        acc_input = layers.Input(
            shape=(self.n_epochs, self.len_acc, 1),
            name='acc_input'
        )

        # ---------------- Epoch encoders ----------------
        ihr_feat = self.epoch_encoder(ihr_input, 'ihr')
        acc_feat = self.epoch_encoder(acc_input, 'acc')

        # ----------------- CoAttention ------------------
        C_D_12 = CoAttention()([ihr_feat, acc_feat])
        C_D_21 = CoAttention()([acc_feat, ihr_feat])

        # ---------------- Fusion ----------------
        x = layers.Concatenate(axis=-1)([C_D_12, C_D_21])
        # x = layers.Concatenate(name='fusion')([ihr_feat, acc_feat])

        # x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        # ---------------- Temporal modeling ----------------
        x = layers.Bidirectional(
            layers.GRU(
                32,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.1
            ),
            name='bigru')(x)

        # ---------------- Output ----------------
        x = layers.TimeDistributed(
            layers.Dense(self.n_classes),
            name='classifier'
        )(x)

        crf = tfa.layers.CRF(3)
        output = crf(logits)

        return tf.keras.Model(
            inputs=[ihr_input, acc_input],
            outputs=outputs,
            name='SleepStage_Small_IHR_ACC'
        )


class SqueezeLayer(layers.Layer):
    def call(self, x):
        # Use your TensorFlow function here
        return tf.squeeze(x, axis=-1)

class ExpandDimLayer(layers.Layer):
    def call(self, x):
        # Use your TensorFlow function here
        return tf.expand_dims(x, axis=-1)

class SleepStageIHRnACC:
    def __init__(self,
                 n_epochs=250,
                 len_ihr=256,
                 len_acc=128,
                 n_classes=3):
        self.n_epochs = n_epochs
        self.len_ihr = len_ihr
        self.len_acc = len_acc
        self.n_classes = n_classes

    def residual_block(self, input, units):
        x = layers.TimeDistributed(layers.Conv1D(units, kernel_size=3, dilation_rate=1, padding="same"))(input)
        x_d_1 = layers.Dense(units)(x)
        x = layers.TimeDistributed(layers.Conv1D(units, kernel_size=3, dilation_rate=2, padding="same"))(x_d_1)
        x_d_2 = layers.Dense(units)(x)
        x = layers.TimeDistributed(layers.Conv1D(units, kernel_size=3, dilation_rate=4, padding="same"))(x_d_2)
        x_d_3 = layers.Dense(units)(x)
        # Skip connection
        if input.shape[-1] != units:
            input = layers.Dense(units)(input)
        x = layers.Concatenate(axis=2)([input, x_d_1, x_d_2, x_d_3])
        x = layers.Dense(units)(x)
        x = layers.Dense(units)(x)
        return x


    def make(self):
        # ---------------- Inputs ----------------
        ihr_input = layers.Input(
            shape=(self.n_epochs, self.len_ihr, 1),
            name='ihr_input'
        )

        acc_input = layers.Input(
            shape=(self.n_epochs, self.len_acc, 1),
            name='acc_input'
        )

        x_norm = layers.BatchNormalization()(ihr_input)
        x_conv = layers.TimeDistributed(layers.Conv1D(8, kernel_size=9, strides=5))(x_norm)

        x = self.residual_block(x_conv, 24)
        x_mp_1 = layers.TimeDistributed(layers.MaxPooling1D(pool_size=5))(x)
        x = self.residual_block(x_mp_1, 48)
        x_mp_2 = layers.TimeDistributed(layers.MaxPooling1D(pool_size=3))(x)
        x = self.residual_block(x_mp_1, 64)
        x_mp_3 = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(x)
        x = self.residual_block(x_mp_1, 91)
        x_mp_4 = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(x)

        y = layers.BatchNormalization()(acc_input)
        y = SqueezeLayer()(y)
        y = layers.Dense(80)(y)
        y = ExpandDimLayer()(y)

        x6 = layers.Concatenate(axis=-1)([y, x_mp_4])

        x1 = layers.TimeDistributed(layers.MaxPooling1D(pool_size=100))(x_norm)
        x2 = layers.TimeDistributed(layers.MaxPooling1D(pool_size=30))(x_mp_1)
        x3 = layers.TimeDistributed(layers.MaxPooling1D(pool_size=12))(x_mp_2)
        x4 = layers.TimeDistributed(layers.MaxPooling1D(pool_size=4))(x_mp_3)
        x5 = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(x_mp_4)
        x = layers.Concatenate(axis=2)([x1, x2, x3, x4, x5, x6])

        x = layers.Dense(192)(x)
        x = layers.Dense(64)(x)

        x_gru1 = layers.Bidirectional(layers.GRU(32))(x)
        x_gru2 = layers.Bidirectional(layers.GRU(32))(x_gru1)
        x_gru3 = layers.Bidirectional(layers.GRU(32))(x_gru2)
        x = layers.Concatenate()([x, x_gru1, x_gru2, x_gru3])

        x = layers.Dense(64)(x)
        x = layers.Dense(16)(x)
        outputs = layers.Dense(4, activation='softmax')(x)

        return tf.keras.Model(
            inputs=[ihr_input, acc_input],
            outputs=outputs,
            name='SleepStage_Small_IHR_ACC'
        )

########################################################################################
import tensorflow as tf
from tensorflow.keras import layers

class PositionalEncoding2(layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def call(self, x):
        seq_len = tf.shape(x)[1]
        pos = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(self.d_model, dtype=tf.float32)[tf.newaxis, :]

        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / self.d_model)
        angle_rads = pos * angle_rates

        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        return x + pos_encoding

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model),
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, training=None):
        attn_output = self.att(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        x1 = self.norm1(x + attn_output)

        ffn_output = self.ffn(x1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(x1 + ffn_output)

class ReshapeLayer(layers.Layer):
    def call(self, x, shape):
        return tf.reshape(x, shape)

class ConcatLayer(layers.Layer):
    def call(self, inputs, axis):
        # The TensorFlow function is used inside the call method
        return tf.concat(inputs, axis=axis)

def build_sleep_model(
    epoch_len=256,      # 30s * 2Hz
    n_epochs=250,      # 2 hours
    n_channels=2,      # IHR + Activity
    d_model=128,
        training=True,
):

    input_hr = tf.keras.Input(shape=(n_epochs, epoch_len, 1))
    input_activity = tf.keras.Input(shape=(n_epochs, epoch_len, 1))
    # input_time = tf.keras.Input(shape=(n_epochs, epoch_len, 1))
    inputs = ConcatLayer()([input_hr, input_activity], axis=-1)
    # inputs = ConcatLayer()([input_hr, input_activity, input_time], axis=-1)

    # =========================
    # 1. LOCAL ENCODER (CNN)
    # =========================
    x = layers.Lambda(lambda t: tf.reshape(t, (-1, epoch_len, n_channels)))(inputs) # (B*240, 60, 2)

    x = layers.Conv1D(32, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)

    x = layers.GlobalAveragePooling1D()(x)  # (B*240, 128)

    # reshape lại sequence
    x = layers.Lambda(lambda t: tf.reshape(t, (-1, n_epochs, 128)))(x) # (B, 240, 128)

    # =========================
    # 2. POSITIONAL ENCODING
    # =========================
    x = PositionalEncoding2(d_model)(x)

    # =========================
    # 3. TRANSFORMER (LONG CONTEXT)
    # =========================
    for _ in range(4):
        x = TransformerBlock(d_model, num_heads=4, ff_dim=256)(x, training=training)
        # x = TransformerBlock(d_model, num_heads=2, ff_dim=256)(x, training=training)

    # =========================
    # 4. CLASSIFIER (per epoch)
    # =========================
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(3, activation="softmax")(x)

    model = tf.keras.Model((input_hr, input_activity), outputs)
    # model = tf.keras.Model((input_hr, input_activity, input_time), outputs)
    return model

from tensorflow.keras import layers

def tcn_block(x, filters, kernel_size=3, dilation_rate=1):
    shortcut = x

    x = layers.Conv1D(
        filters,
        kernel_size,
        padding='same',
        dilation_rate=dilation_rate,
        activation='relu'
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(
        filters,
        kernel_size,
        padding='same',
        dilation_rate=dilation_rate,
        activation='relu'
    )(x)

    x = layers.BatchNormalization()(x)

    # residual
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)

    x = layers.Add()([x, shortcut])
    return x

def build_sleep_model_v2(
    epoch_len=256,
    n_epochs=250,
    n_channels=4,   # 3 IHR + 1 ACC
):

    # inputs = tf.keras.Input(shape=(n_epochs, epoch_len, n_channels))

    input_hr = tf.keras.Input(shape=(n_epochs, epoch_len, 1))
    input_activity = tf.keras.Input(shape=(n_epochs, epoch_len, 1))
    # input_time = tf.keras.Input(shape=(n_epochs, epoch_len, 1))
    inputs = ConcatLayer()([input_hr, input_activity], axis=-1)

    # =========================
    # 1. LOCAL CNN (per epoch)
    # =========================
    x = layers.TimeDistributed(
        layers.Conv1D(32, 5, padding="same", activation="relu")
    )(inputs)

    x = layers.TimeDistributed(layers.BatchNormalization())(x)

    x = layers.TimeDistributed(
        layers.Conv1D(64, 5, padding="same", activation="relu")
    )(x)

    x = layers.TimeDistributed(layers.BatchNormalization())(x)

    x = layers.TimeDistributed(
        layers.Conv1D(128, 3, padding="same", activation="relu")
    )(x)

    x = layers.TimeDistributed(
        layers.GlobalAveragePooling1D()
    )(x)

    # shape: (B, T, 128)

    # =========================
    # 2. TCN (temporal)
    # =========================
    # dilation tăng dần → học long context
    x = tcn_block(x, 128, dilation_rate=1)
    x = tcn_block(x, 128, dilation_rate=2)
    x = tcn_block(x, 128, dilation_rate=4)
    x = tcn_block(x, 128, dilation_rate=8)

    # =========================
    # 3. CLASSIFIER
    # =========================
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(3, activation="softmax")(x)

    return tf.keras.Model((input_hr, input_activity), outputs)

class TCNBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, dilation_rate=1, dropout=0.1):
        super().__init__()
        self.conv1 = layers.Conv1D(
            filters, kernel_size,
            padding="causal",
            dilation_rate=dilation_rate,
            activation="relu"
        )
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv1D(
            filters, kernel_size,
            padding="causal",
            dilation_rate=dilation_rate,
            activation="relu"
        )
        self.bn2 = layers.BatchNormalization()

        self.dropout = layers.Dropout(dropout)

        self.downsample = layers.Conv1D(filters, 1, padding="same")

    def call(self, x, training=None):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.dropout(x, training=training)

        # match dimension
        if residual.shape[-1] != x.shape[-1]:
            residual = self.downsample(residual)

        return tf.nn.relu(x + residual)

def build_sleep_model_tcn(
    epoch_len=256,
    n_epochs=250,
    n_channels=2,
    d_model=128,
):

    input_hr = tf.keras.Input(shape=(n_epochs, epoch_len, 1))
    input_activity = tf.keras.Input(shape=(n_epochs, epoch_len, 1))

    inputs = ConcatLayer()([input_hr, input_activity], axis=-1)

    # =========================
    # 1. LOCAL ENCODER (TCN)
    # =========================
    x = layers.Lambda(lambda t: tf.reshape(t, (-1, epoch_len, n_channels)))(inputs)  # (B*epochs, 256, 2)

    # TCN stack (multi dilation)
    for d in [1, 2, 4, 8]:
        x = TCNBlock(64, kernel_size=5, dilation_rate=d)(x)

    x = layers.Conv1D(d_model, 1, activation="relu")(x)

    # ❗ giữ temporal rồi mới pooling
    x = layers.GlobalAveragePooling1D()(x)

    # reshape lại sequence
    x = layers.Lambda(lambda t: tf.reshape(t, (-1, n_epochs, d_model)))(x)  # (B, epochs, d_model)

    # =========================
    # 2. POSITIONAL ENCODING
    # =========================
    x = PositionalEncoding2(d_model)(x)

    # =========================
    # 3. TRANSFORMER
    # =========================
    for _ in range(4):
        x = TransformerBlock(d_model, num_heads=4, ff_dim=256)(x)

    # =========================
    # 4. CLASSIFIER
    # =========================
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(3, activation="softmax")(x)

    model = tf.keras.Model((input_hr, input_activity), outputs)

    return model

def create_local_mask(seq_len, window_size=5):
    mask = tf.ones((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            if abs(i - j) > window_size:
                mask = tf.tensor_scatter_nd_update(
                    mask, [[i, j]], [0.0]
                )
    return mask

class TransformerBlockMS(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model),
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, training=None, mask=None):
        attn_output = self.att(x, x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        x1 = self.norm1(x + attn_output)

        ffn_output = self.ffn(x1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(x1 + ffn_output)


class FlatAttentionFusion(layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # Shared attention params
        self.Wa = layers.Dense(d_model)
        self.va = layers.Dense(1)

        # Encoder-specific projections U_c^(k)
        self.Uc_short = layers.Dense(d_model)
        self.Uc_long = layers.Dense(d_model)

    def call(self, short_x, long_x):
        # short_x, long_x: (B, T, D)

        # Project to shared space
        short_proj = self.Uc_short(short_x)   # (B, T, D)
        long_proj  = self.Uc_long(long_x)     # (B, T, D)

        # Compute energy
        e_short = self.va(tf.nn.tanh(self.Wa(short_x)))  # (B, T, 1)
        e_long  = self.va(tf.nn.tanh(self.Wa(long_x)))   # (B, T, 1)

        # Concatenate along time (joint normalization!)
        e_all = tf.concat([e_short, e_long], axis=1)  # (B, 2T, 1)

        alpha = tf.nn.softmax(e_all, axis=1)  # joint softmax

        # Split alpha
        alpha_short, alpha_long = tf.split(alpha, 2, axis=1)

        # Weighted sum
        c_short = alpha_short * short_proj
        c_long  = alpha_long  * long_proj

        # Final context
        c = tf.concat([c_short, c_long], axis=-1)  # (B, T, D)

        return c

def build_sleep_model_multiscale(
    epoch_len=256,
    n_epochs=250,
    n_channels=2,
    d_model=128,
        training=False
):

    input_hr = tf.keras.Input(shape=(n_epochs, epoch_len, 1))
    input_activity = tf.keras.Input(shape=(n_epochs, epoch_len, 1))

    x = ConcatLayer()([input_hr, input_activity], axis=-1)

    # =========================
    # 1. TCN encoder (local signal)
    # =========================
    x = layers.Lambda(lambda t: tf.reshape(t, (-1, epoch_len, n_channels)))(x)

    for d in [1, 2, 4, 8]:
        x = TCNBlock(64, kernel_size=5, dilation_rate=d)(x)

    x = layers.Conv1D(d_model, 1, activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)

    # x = layers.Lambda(lambda t: tf.reshape(t, (-1, epoch_len, d_model)))(x)  # (B, T, D)
    x = layers.Lambda(lambda t: tf.reshape(t, (-1, n_epochs, d_model)))(x)
    x = PositionalEncoding2(d_model)(x)

    # =========================
    # 2. SHORT CONTEXT BRANCH
    # =========================
    short_x = x

    seq_len = n_epochs
    local_mask = create_local_mask(seq_len, window_size=5)

    for _ in range(2):
        short_x = TransformerBlockMS(
            d_model, num_heads=2, ff_dim=128
        )(short_x, mask=local_mask, training=training)

    # =========================
    # 3. LONG CONTEXT BRANCH
    # =========================
    long_x = x

    for _ in range(4):
        long_x = TransformerBlockMS(
            d_model, num_heads=4, ff_dim=256
        )(long_x)

    # =========================
    # 4. FUSION
    # =========================
    x = ConcatLayer()([short_x, long_x], axis=-1)
    # x = FlatAttentionFusion(d_model)(short_x, long_x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    # =========================
    # 5. CLASSIFIER
    # =========================
    outputs = layers.Dense(3, activation="softmax")(x)

    model = tf.keras.Model((input_hr, input_activity), outputs)

    return model

class TCNBlockV2(layers.Layer):
    def __init__(self, filters,
                 kernel_size=3,
                 dilation_rate=1,
                 dropout=0.1):

        super().__init__()

        self.conv1 = layers.Conv1D(
            filters,
            kernel_size,
            padding="causal",
            dilation_rate=dilation_rate,
            activation="relu"
        )

        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv1D(
            filters,
            kernel_size,
            padding="causal",
            dilation_rate=dilation_rate,
            activation="relu"
        )

        self.bn2 = layers.BatchNormalization()

        self.dropout = layers.Dropout(dropout)

        self.downsample = layers.Conv1D(filters, 1, padding="same")

    def call(self, x, training=None):

        residual = x

        x = self.conv1(x)
        x = self.bn1(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.dropout(x, training=training)

        if residual.shape[-1] != x.shape[-1]:
            residual = self.downsample(residual)

        return tf.nn.relu(x + residual)

def build_sleep_model_multiscale_v2(
    epoch_len=256,
    n_epochs=250,
    n_channels=2,
    d_model=128,
    training=False
):

    input_hr = tf.keras.Input(
        shape=(n_epochs, epoch_len, 1)
    )

    input_activity = tf.keras.Input(
        shape=(n_epochs, epoch_len, 1)
    )

    x = ConcatLayer()(
        [input_hr, input_activity],
        axis=-1
    )

    # ==================================================
    # reshape
    # ==================================================
    # x = layers.Reshape(
    #     (n_epochs * epoch_len, n_channels)
    # )(x)
    #
    # x = layers.Reshape(
    #     (-1, epoch_len, n_channels)
    # )(x)

    x = layers.Lambda(lambda t: tf.reshape(t, (-1, epoch_len, n_channels)))(x)

    # ==================================================
    # SHORT TEMPORAL BRANCH
    # preserve wake spikes
    # ==================================================
    short_x = x

    for d in [1, 2]:
        short_x = TCNBlockV2(
            64,
            kernel_size=3,
            dilation_rate=d
        )(short_x)

    short_x = layers.Conv1D(
        d_model,
        1,
        activation="relu"
    )(short_x)

    # IMPORTANT
    short_x = layers.GlobalMaxPooling1D()(short_x)

    short_x = layers.Lambda(lambda t: tf.reshape(t, (-1, n_epochs, d_model)))(short_x)

    short_x = PositionalEncoding2(d_model)(short_x)

    # ==================================================
    # LONG TEMPORAL BRANCH
    # sleep structure
    # ==================================================
    long_x = x

    for d in [4, 8]:
        long_x = TCNBlockV2(
            64,
            kernel_size=5,
            dilation_rate=d
        )(long_x)

    long_x = layers.Conv1D(
        d_model,
        1,
        activation="relu"
    )(long_x)

    long_x = layers.GlobalAveragePooling1D()(long_x)

    long_x = layers.Lambda(lambda t: tf.reshape(t, (-1, n_epochs, d_model)))(long_x)

    long_x = PositionalEncoding2(d_model)(long_x)

    # ==================================================
    # SHORT CONTEXT TRANSFORMER
    # ==================================================
    seq_len = n_epochs

    local_mask = create_local_mask(
        seq_len,
        window_size=5
    )

    for _ in range(2):
        short_x = TransformerBlockMS(
            d_model,
            num_heads=2,
            ff_dim=128
        )(
            short_x,
            mask=local_mask,
            training=training
        )

    # ==================================================
    # LONG CONTEXT TRANSFORMER
    # ==================================================
    for _ in range(4):
        long_x = TransformerBlockMS(
            d_model,
            num_heads=4,
            ff_dim=256
        )(
            long_x,
            training=training
        )

    # ==================================================
    # FUSION
    # ==================================================
    x = layers.Concatenate(axis=-1)(
        [short_x, long_x]
    )

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(
        3,
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        [input_hr, input_activity],
        outputs
    )

    return model

def build_sleep_model_multiscale_v3(
    epoch_len=256,
    n_epochs=250,
    n_channels=2,
    d_model=128,
        training=False
):

    input_hr = tf.keras.Input(shape=(n_epochs, epoch_len, 1))
    input_activity = tf.keras.Input(shape=(n_epochs, epoch_len, 1))

    x = ConcatLayer()([input_hr, input_activity], axis=-1)

    # =========================
    # 1. TCN encoder (local signal)
    # =========================
    x = layers.Lambda(lambda t: tf.reshape(t, (-1, epoch_len, n_channels)))(x)

    for d in [1, 2, 4, 8]:
        x = TCNBlock(64, kernel_size=5, dilation_rate=d)(x)

    x = layers.Conv1D(d_model, 1, activation="relu")(x)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)

    avg_pool = layers.Dense(d_model)(avg_pool)
    max_pool = layers.Dense(d_model)(max_pool)

    gate = layers.Dense(
        d_model,
        activation="sigmoid"
    )(max_pool)

    x = avg_pool + gate * max_pool

    # x = layers.Lambda(lambda t: tf.reshape(t, (-1, epoch_len, d_model)))(x)  # (B, T, D)
    x = layers.Lambda(lambda t: tf.reshape(t, (-1, n_epochs, d_model)))(x)
    x = PositionalEncoding2(d_model)(x)

    # =========================
    # 2. SHORT CONTEXT BRANCH
    # =========================
    short_x = x

    seq_len = n_epochs
    local_mask = create_local_mask(seq_len, window_size=5)

    for _ in range(2):
        short_x = TransformerBlockMS(
            d_model, num_heads=2, ff_dim=128
        )(short_x, mask=local_mask, training=training)

    # =========================
    # 3. LONG CONTEXT BRANCH
    # =========================
    long_x = x

    for _ in range(4):
        long_x = TransformerBlockMS(
            d_model, num_heads=4, ff_dim=256
        )(long_x)

    # =========================
    # 4. FUSION
    # =========================
    x = ConcatLayer()([short_x, long_x], axis=-1)
    # x = FlatAttentionFusion(d_model)(short_x, long_x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    # =========================
    # 5. CLASSIFIER
    # =========================
    outputs = layers.Dense(3, activation="softmax")(x)

    model = tf.keras.Model((input_hr, input_activity), outputs)

    return model

def build_sleep_model_multiscale_v4(
    epoch_len=256,
    n_epochs=250,
    n_channels=2,
    d_model=128,
    training=False
):

    # =====================================================
    # INPUT
    # =====================================================
    input_hr = tf.keras.Input(shape=(n_epochs, epoch_len, 1))
    input_activity = tf.keras.Input(shape=(n_epochs, epoch_len, 1))

    x = ConcatLayer()([input_hr, input_activity], axis=-1)

    # =====================================================
    # RESHAPE
    # (B, T, L, C) -> (B*T, L, C)
    # =====================================================
    x = layers.Lambda(lambda t: tf.reshape(t, (-1, epoch_len, n_channels)))(x)

    # =====================================================
    # 1. MAIN TCN ENCODER
    # =====================================================
    main_x = x

    for d in [1, 2, 4, 8]:
        main_x = TCNBlock(
            64,
            kernel_size=5,
            dilation_rate=d
        )(main_x)

    main_x = layers.Conv1D(
        d_model,
        1,
        activation="relu"
    )(main_x)

    # =====================================================
    # MAIN POOLING
    # =====================================================
    avg_pool = layers.GlobalAveragePooling1D()(main_x)

    main_x = layers.Lambda(lambda t: tf.reshape(t, (-1, n_epochs, d_model)))(avg_pool)

    # =====================================================
    # 2. WAKE TRANSIENT BRANCH
    # =====================================================
    wake_x = x

    # ----- kernel nhỏ -----
    wake_x1 = layers.Conv1D(
        32,
        3,
        padding="same",
        activation="relu"
    )(wake_x)

    # ----- kernel trung bình -----
    wake_x2 = layers.Conv1D(
        32,
        5,
        padding="same",
        activation="relu"
    )(wake_x)

    # # ----- kernel lớn hơn -----
    # wake_x3 = layers.Conv1D(
    #     32,
    #     9,
    #     padding="same",
    #     activation="relu"
    # )(wake_x)

    wake_x = layers.Concatenate()([
        wake_x1,
        wake_x2,
        # wake_x3
    ])

    wake_x = layers.Conv1D(d_model, 1, activation="relu")(wake_x)
    # wake transient rất hợp max pooling
    wake_x = layers.GlobalMaxPooling1D()(wake_x)


    # =====================================================
    # 3. GATED WAKE FUSION
    # =====================================================
    wake_x = layers.Lambda(lambda t: tf.reshape(t, (-1, n_epochs, d_model)))(wake_x)

    # wake branch influence nhỏ
    fused_x = layers.Add()([main_x, wake_x])

    # =====================================================
    # RESHAPE BACK TO SEQUENCE
    # =====================================================
    x = layers.Reshape(
        (n_epochs, d_model)
    )(fused_x)

    x = PositionalEncoding2(d_model)(x)

    # =====================================================
    # 4. SHORT CONTEXT BRANCH
    # =====================================================
    short_x = x

    seq_len = n_epochs

    local_mask = create_local_mask(
        seq_len,
        window_size=5
    )

    for _ in range(2):
        short_x = TransformerBlockMS(
            d_model,
            num_heads=2,
            ff_dim=128
        )(
            short_x,
            mask=local_mask,
            training=training
        )

    # =====================================================
    # 5. LONG CONTEXT BRANCH
    # =====================================================
    long_x = x

    for _ in range(4):
        long_x = TransformerBlockMS(
            d_model,
            num_heads=4,
            ff_dim=256
        )(
            long_x,
            training=training
        )

    # =====================================================
    # 6. FINAL FUSION
    # =====================================================
    x = ConcatLayer()([
        short_x,
        long_x
    ], axis=-1)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.2)(x)

    # =====================================================
    # 7. CLASSIFIER
    # =====================================================
    outputs = layers.Dense(
        3,
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        (input_hr, input_activity),
        outputs
    )

    return model

def build_sleep_model_multiscale_v5(
    epoch_len=256,
    n_epochs=250,
    n_channels=2,
    d_model=128,
        training=False
):

    input_hr = tf.keras.Input(shape=(n_epochs, epoch_len, 1))
    input_activity = tf.keras.Input(shape=(n_epochs, epoch_len, 1))

    x = ConcatLayer()([input_hr, input_activity], axis=-1)

    # =========================
    # 1. TCN encoder (local signal)
    # =========================
    x = layers.Lambda(lambda t: tf.reshape(t, (-1, epoch_len, n_channels)))(x)

    for d in [1, 2, 4, 8]:
        x = TCNBlock(64, kernel_size=3, dilation_rate=d)(x)

    x = layers.Conv1D(d_model, 1, activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)

    # x = layers.Lambda(lambda t: tf.reshape(t, (-1, epoch_len, d_model)))(x)  # (B, T, D)
    x = layers.Lambda(lambda t: tf.reshape(t, (-1, n_epochs, d_model)))(x)
    x = PositionalEncoding2(d_model)(x)

    # =========================
    # 2. SHORT CONTEXT BRANCH
    # =========================
    short_x = x

    seq_len = n_epochs
    local_mask = create_local_mask(seq_len, window_size=5)

    for _ in range(2):
        short_x = TransformerBlockMS(
            d_model, num_heads=2, ff_dim=128
        )(short_x, mask=local_mask, training=training)

    # =========================
    # 3. LONG CONTEXT BRANCH
    # =========================
    long_x = x

    for _ in range(4):
        long_x = TransformerBlockMS(
            d_model, num_heads=4, ff_dim=256
        )(long_x)

    # =========================
    # 4. FUSION
    # =========================
    x = ConcatLayer()([short_x, long_x], axis=-1)
    # x = FlatAttentionFusion(d_model)(short_x, long_x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    # =========================
    # 5. CLASSIFIER
    # =========================
    outputs = layers.Dense(3, activation="softmax")(x)

    model = tf.keras.Model((input_hr, input_activity), outputs)

    return model

def build_sleep_model_multiscale_v6(
    epoch_len=256,
    n_epochs=250,
    n_channels=2,
    d_model=128,
        training=False
):

    input_hr = tf.keras.Input(shape=(n_epochs, epoch_len, 1))
    input_activity = tf.keras.Input(shape=(n_epochs, epoch_len, 1))

    x = ConcatLayer()([input_hr, input_activity], axis=-1)

    # =========================
    # 1. TCN encoder (local signal)
    # =========================
    x = layers.Lambda(lambda t: tf.reshape(t, (-1, epoch_len, n_channels)))(x)

    for d in [1, 2, 4, 8]:
        x = TCNBlock(64, kernel_size=5, dilation_rate=d)(x)

    x = layers.Conv1D(d_model, 1, activation="relu")(x)

    x = layers.GlobalAveragePooling1D()(x)

    # x = layers.Lambda(lambda t: tf.reshape(t, (-1, epoch_len, d_model)))(x)  # (B, T, D)
    x = layers.Lambda(lambda t: tf.reshape(t, (-1, n_epochs, d_model)))(x)
    x = PositionalEncoding2(d_model)(x)

    act = layers.Lambda(
        lambda t: tf.reshape(t, (-1, epoch_len, 1))
    )(input_activity)

    act = layers.Conv1D(
        32,
        kernel_size=5,
        padding="same",
        activation="relu"
    )(act)

    act = layers.Conv1D(
        64,
        kernel_size=5,
        padding="same",
        activation="relu"
    )(act)

    # quan trọng
    act_gap = layers.GlobalAveragePooling1D()(act)
    act_gmp = layers.GlobalMaxPooling1D()(act)

    act = layers.Concatenate()([
        act_gap,
        act_gmp
    ])

    act = layers.Dense(
        d_model,
        activation="relu"
    )(act)

    local_feat = layers.Lambda(
        lambda t: tf.reshape(t, (-1, n_epochs, d_model))
    )(act)

    # local_feat = x

    # =========================
    # 2. SHORT CONTEXT BRANCH
    # =========================
    short_x = x

    seq_len = n_epochs
    local_mask = create_local_mask(seq_len, window_size=5)

    for _ in range(2):
        short_x = TransformerBlockMS(
            d_model, num_heads=2, ff_dim=128
        )(short_x, mask=local_mask, training=training)

    # =========================
    # 3. LONG CONTEXT BRANCH
    # =========================
    long_x = x

    for _ in range(4):
        long_x = TransformerBlockMS(
            d_model, num_heads=4, ff_dim=256
        )(long_x)

    # =========================
    # 4. FUSION
    # =========================
    # x = short_x
    # x = ConcatLayer()([short_x, long_x], axis=-1)
    x = ConcatLayer()([local_feat, short_x, long_x], axis=-1)
    # x = FlatAttentionFusion(d_model)(short_x, long_x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    # =========================
    # 5. CLASSIFIER
    # =========================
    outputs = layers.Dense(3, activation="softmax")(x)

    model = tf.keras.Model((input_hr, input_activity), outputs)

    return model

def build_sleep_model_multiscale_v7(
    epoch_len=256,
    n_epochs=250,
    n_channels=2,
    d_model=128,
        training=False
):

    input_hr = tf.keras.Input(shape=(n_epochs, epoch_len, 1))
    input_activity = tf.keras.Input(shape=(n_epochs, epoch_len, 1))

    x = ConcatLayer()([input_hr, input_activity], axis=-1)

    # =========================
    # 1. TCN encoder (local signal)
    # =========================
    x = layers.Lambda(lambda t: tf.reshape(t, (-1, epoch_len, n_channels)))(x)

    for d in [1, 2, 4, 8]:
        x = TCNBlock(64, kernel_size=5, dilation_rate=d)(x)

    x = layers.Conv1D(d_model, 1, activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)

    # x = layers.Lambda(lambda t: tf.reshape(t, (-1, epoch_len, d_model)))(x)  # (B, T, D)
    x = layers.Lambda(lambda t: tf.reshape(t, (-1, n_epochs, d_model)))(x)
    # x = PositionalEncoding2(d_model)(x)

    # =========================
    # 2. SHORT CONTEXT BRANCH
    # =========================
    short_x = x

    seq_len = n_epochs
    local_mask = create_local_mask(seq_len, window_size=5)

    for _ in range(2):
        short_x = TransformerBlockMS(
            d_model, num_heads=2, ff_dim=128
        )(short_x, mask=local_mask, training=training)

    # =========================
    # 3. LONG CONTEXT BRANCH
    # =========================
    long_x = x

    for _ in range(4):
        long_x = TransformerBlockMS(
            d_model, num_heads=4, ff_dim=256
        )(long_x)

    # =========================
    # 4. FUSION
    # =========================
    x = ConcatLayer()([short_x, long_x], axis=-1)
    # x = FlatAttentionFusion(d_model)(short_x, long_x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    # =========================
    # 5. CLASSIFIER
    # =========================
    outputs = layers.Dense(3, activation="softmax")(x)

    model = tf.keras.Model((input_hr, input_activity), outputs)

    return model

# transition matrix (prior knowledge)
T = [
  # NREM, REM, Wake
  [0.9, 0.1, 0.0],
  [0.2, 0.7, 0.1],
  [0.3, 0.0, 0.7],
]

def transition_loss(y_pred):
    p_t = y_pred[:, :-1, :]
    p_t1 = y_pred[:, 1:, :]

    trans_prob = tf.matmul(p_t, T)
    loss = tf.reduce_mean((p_t1 - trans_prob)**2)

    return loss

def weighted_ce_loss(class_weights):
    class_weights = tf.constant(class_weights, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        # y_true: (B, 240)
        # y_pred: (B, 240, 3)

        # y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=3)

        weights = tf.reduce_sum(class_weights * y_true, axis=-1)

        ce = tf.keras.losses.categorical_crossentropy(
            y_true, y_pred
        )

        return tf.reduce_mean(ce * weights)

    return loss_fn

def temporal_kl_loss(y_pred):
    p = y_pred[:, :-1, :]
    q = y_pred[:, 1:, :]

    return tf.reduce_mean(
        tf.reduce_sum(p * tf.math.log((p+1e-8)/(q+1e-8)), axis=-1)
    )

def temporal_smoothness_loss(y_pred):
    # penalize difference between consecutive predictions
    diff = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    return tf.reduce_mean(tf.square(diff))

# def temporal_smoothness_loss(y_pred, lambda_smooth_wake=0.05, lambda_smooth=0.1):
#     # penalize difference between consecutive predictions
#     sleep_prob = y_pred[..., 1:]
#     diff = sleep_prob[:, 1:, :] - sleep_prob[:, :-1, :]
#
#     wake_prob = y_pred[..., 0]
#     wake_diff = wake_prob[:, 1:] - wake_prob[:, :-1]
#     return lambda_smooth*tf.reduce_mean(tf.square(diff)) + lambda_smooth_wake*tf.reduce_mean(tf.square(wake_diff))


def temporal_smoothness_loss_nremrem(y_pred):
    # only REM/NREM
    sleep_prob = y_pred[..., 1:]
    diff = sleep_prob[:, 1:, :] - sleep_prob[:, :-1, :]
    return tf.reduce_mean(tf.square(diff))

def combined_loss(class_weights, lambda_smooth=0.1, lambda_smooth_wake=0.05):
    ce = weighted_ce_loss(class_weights)

    def loss(y_true, y_pred):
        # return ce(y_true, y_pred) + lambda_smooth * temporal_smoothness_loss_nremrem(y_pred) # Version 18k
        # return ce(y_true, y_pred) + temporal_smoothness_loss(y_pred, lambda_smooth=lambda_smooth, lambda_smooth_wake=lambda_smooth_wake) # Version 18l
        return ce(y_true, y_pred) + lambda_smooth * temporal_smoothness_loss(y_pred) # Version 18
        # return ce(y_true, y_pred) + lambda_smooth * temporal_kl_loss(y_pred) # Version 18b
        # return ce(y_true, y_pred) + lambda_smooth * temporal_smoothness_loss(y_pred) + 0.2 * transition_loss(y_pred) # Version 18c

    return loss

def weighted_focal_loss(class_weights, gamma=2.0):

    alpha = tf.constant(
        class_weights,
        dtype=tf.float32
    )

    def loss(y_true, y_pred):

        y_pred = tf.clip_by_value(
            y_pred,
            1e-7,
            1.0 - 1e-7
        )

        ce = -y_true * tf.math.log(y_pred)

        focal_factor = tf.pow(
            1.0 - y_pred,
            gamma
        )

        alpha_factor = alpha * y_true

        loss = alpha_factor * focal_factor * ce

        return tf.reduce_mean(
            tf.reduce_sum(loss, axis=-1)
        )

    return loss

def combined_loss_focal_loss(
    class_weights,
    gamma=2.0,
    lambda_smooth=0.1
):

    focal = weighted_focal_loss(
        class_weights,
        gamma
    )

    def loss(y_true, y_pred):

        focal_loss = focal(
            y_true,
            y_pred
        )

        smooth_loss = temporal_smoothness_loss(
            y_pred
        )

        return (
            focal_loss
            + lambda_smooth * smooth_loss
        )

    return loss

def sample_sequences(y_pred, num_samples=5):
    # y_pred: (B, T, C)

    B, T, C = tf.shape(y_pred)[0], tf.shape(y_pred)[1], tf.shape(y_pred)[2]

    # reshape for sampling
    logits = tf.math.log(y_pred + 1e-8)

    samples = tf.random.categorical(
        tf.reshape(logits, [-1, C]), num_samples
    )  # (B*T, N)

    samples = tf.reshape(samples, [B, T, num_samples])  # (B, T, N)

    return samples

def sequence_error(y_true, y_sample):
    # y_true: (B, T)
    # y_sample: (B, T)

    return tf.reduce_sum(
        tf.cast(tf.not_equal(y_true, y_sample), tf.float32),
        axis=1  # sum over time
    )  # (B,)

def mwer_loss(lambda_ce=0.3, num_samples=5):

    ce_loss_fn = tf.keras.losses.CategoricalCrossentropy()

    def loss(y_true, y_pred):
        # y_true: (B, T, C) one-hot
        # y_pred: (B, T, C)

        y_true_ids = tf.argmax(y_true, axis=-1)  # (B, T)

        samples = sample_sequences(y_pred, num_samples)  # (B, T, N)

        losses = []
        probs = []

        for i in range(num_samples):
            y_i = samples[:, :, i]  # (B, T)

            # compute error
            W_i = sequence_error(y_true_ids, y_i)  # (B,)

            # compute log prob
            y_i_onehot = tf.one_hot(y_i, depth=tf.shape(y_pred)[-1])
            prob = tf.reduce_sum(
                y_i_onehot * tf.math.log(y_pred + 1e-8),
                axis=[1, 2]
            )  # (B,)

            losses.append(W_i)
            probs.append(prob)

        W = tf.stack(losses, axis=1)  # (B, N)
        logP = tf.stack(probs, axis=1)  # (B, N)

        # normalize probability
        P_hat = tf.nn.softmax(logP, axis=1)

        # variance reduction
        W_bar = tf.reduce_mean(W, axis=1, keepdims=True)

        # MWER
        mwer = tf.reduce_sum(
            P_hat * (W - W_bar),
            axis=1
        )  # (B,)

        # CE (stabilization)
        ce = ce_loss_fn(y_true, y_pred)

        return tf.reduce_mean(mwer) + lambda_ce * ce

    return loss

def final_loss(class_weights, lambda_ce=0.3, lambda_smooth=0.1):

    mwer = mwer_loss(lambda_ce=lambda_ce)

    def loss(y_true, y_pred):
        return (
            mwer(y_true, y_pred)
            + lambda_smooth * temporal_smoothness_loss(y_pred)
        )

    return loss


if __name__ == '__main__':
    # model = SleepStageIHRnACCModel(299, 256, 128, 3).make()
    # model = SleepStageSmallIHR_ACC_CoAttention(250, 256, 128, 3).make()
    # model = ImprovedSleepStageSmallIHR_ACC_CoAttention(250, 256, 128, 3).make()
    # model = ImprovedSleepStageSmallIHR_ACC_CoAttention(250, 256, 256, 3).make()
    # model = ImprovedSleepStageSmallIHR_ACC_CoAttention2(250, 256, 256, 3).make()
    # model = SleepStageIHRnACC(250, 256, 128, 3).make()
    # model = build_sleep_model(epoch_len=256,      # 30s * 2Hz
    # n_epochs=250,      # 2 hours
    # n_channels=2,      # IHR + Activity
    # d_model=128)

    # model = build_sleep_model_v2(epoch_len=256,  # 30s * 2Hz
    #                           n_epochs=250,  # 2 hours
    #                              )
    # model = build_sleep_model_tcn(
    #                                 epoch_len=256,
    #                                 n_epochs=250,
    #                                 n_channels=2,
    #                                 d_model=128,
    #                             )
    # model = build_sleep_model_multiscale(epoch_len=256,
    #                                      n_epochs=250,
    #                                      n_channels=2,
    #                                      d_model=128,
    #                                      training=True)

    # model = build_sleep_model_multiscale_v2(epoch_len=256,
    #                                      n_epochs=250,
    #                                      n_channels=2,
    #                                      d_model=128,
    #                                      training=True)

    model = build_sleep_model_multiscale_v7(epoch_len=256,
                                            n_epochs=250,
                                            n_channels=2,
                                            d_model=128,
                                            training=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=combined_loss([1.5, 2.0, 1.0]),
        # loss=final_loss([1.5, 2.0, 1.0]),
        metrics=["accuracy"]
    )

    model.summary()

    import numpy as np
    from tensorflow.keras.utils import to_categorical

    # X = np.random.randn(8, 250, 256, 2).astype(np.float32)
    # y = np.random.randint(0, 3, (8, 250))
    #
    # model.fit(X, y, epochs=10)

    X = np.random.randn(8, 250, 256, 1).astype(np.float32)
    X_acc = np.random.randn(8, 250, 256, 1).astype(np.float32)
    # X_hrv = np.random.randn(8, 250, 4, 1).astype(np.float32)
    # X_time = np.random.randn(8, 250, 256, 1).astype(np.float32)
    y = np.random.randint(0, 3, (8, 250))
    y = to_categorical(y, num_classes=3)
    print(y.shape)

    model.fit((X, X_acc), y, epochs=10)