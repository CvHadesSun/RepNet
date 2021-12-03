import tensorflow.compat.v2 as tf
from .utils import *
from .transformer import TransformerLayer


# Model definition
layers = tf.keras.layers
regularizers = tf.keras.regularizers


class NoFeatureExtractPeriodEstimator(tf.keras.models.Model):
    """RepNet model,which has no feature extractor network."""
    def __init__(
            self,
            num_frames=64,
            image_size=112,
            dropout_rate=0.25,
            l2_reg_weight=1e-6,
            conv_channels=32,
            conv_kernel_size=3,
            transformer_layers_config=((512, 4, 512),),
            transformer_dropout_rate=0.0,
            transformer_reorder_ln=True,
            period_fc_channels=(512, 512),
            within_period_fc_channels=(512, 512)):
        super(NoFeatureExtractPeriodEstimator, self).__init__()

        
        self.num_frames = num_frames
        self.image_size = image_size

        self.dropout_rate = dropout_rate
        self.l2_reg_weight = l2_reg_weight
        
        self.conv_channels = conv_channels
        self.conv_kernel_size = conv_kernel_size
        self.transformer_layers_config = transformer_layers_config
        # Transformer config in form of (channels, heads, bottleneck channels).
        self.transformer_layers_config = transformer_layers_config
        self.transformer_dropout_rate = transformer_dropout_rate
        self.transformer_reorder_ln = transformer_reorder_ln
        self.period_fc_channels = period_fc_channels
        self.within_period_fc_channels = within_period_fc_channels


        # Counting Module (Self-sim > Conv > Transformer > Classifier)
        self.conv_3x3_layer = layers.Conv2D(self.conv_channels,
                                            self.conv_kernel_size,
                                            padding='same',
                                        activation=tf.nn.relu)

        channels = self.transformer_layers_config[0][0]
        self.input_projection = layers.Dense(
            channels, kernel_regularizer=regularizers.l2(self.l2_reg_weight),
            activation=None)
        self.input_projection2 = layers.Dense(
            channels, kernel_regularizer=regularizers.l2(self.l2_reg_weight),
            activation=None)

        length = self.num_frames
        self.pos_encoding = tf.compat.v1.get_variable(
            name='resnet_period_estimator/pos_encoding',
            shape=[1, length, 1],
            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))
        self.pos_encoding2 = tf.compat.v1.get_variable(
            name='resnet_period_estimator/pos_encoding2',
            shape=[1, length, 1],
            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))

        
        self.transformer_layers = []
        for d_model, num_heads, dff in self.transformer_layers_config:
            self.transformer_layers.append(
                        TransformerLayer(d_model, num_heads, dff,
                           self.transformer_dropout_rate,
                           self.transformer_reorder_ln))

        self.transformer_layers2 = []
        for d_model, num_heads, dff in self.transformer_layers_config:
            self.transformer_layers2.append(
                        TransformerLayer(d_model, num_heads, dff,
                           self.transformer_dropout_rate,
                           self.transformer_reorder_ln))

        # Period Prediction Module.
        self.dropout_layer = layers.Dropout(self.dropout_rate)
        num_preds = self.num_frames//2
        self.fc_layers = []
        for channels in self.period_fc_channels:
            self.fc_layers.append(layers.Dense(
                    channels, kernel_regularizer=regularizers.l2(self.l2_reg_weight),
                    activation=tf.nn.relu))
        self.fc_layers.append(layers.Dense(
                num_preds, kernel_regularizer=regularizers.l2(self.l2_reg_weight)))

        # Within Period Module
        num_preds = 1
        self.within_period_fc_layers = []
        for channels in self.within_period_fc_channels:
            self.within_period_fc_layers.append(layers.Dense(
                channels, kernel_regularizer=regularizers.l2(self.l2_reg_weight),
                activation=tf.nn.relu))
        self.within_period_fc_layers.append(layers.Dense(
                num_preds, kernel_regularizer=regularizers.l2(self.l2_reg_weight)))

    def call(self,x):
        batch_size = tf.shape(x)[0]  # directly is affinity matrix input 
        affinity=x

        # 3x3 conv layer on self-similarity matrix.
        x = self.conv_3x3_layer(x)
        x = tf.reshape(x, [batch_size, self.num_frames, -1])
        within_period_x = x


        # Period prediction.
        x = self.input_projection(x)
        x += self.pos_encoding
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        x = flatten_sequential_feats(x, batch_size, self.num_frames)
        for fc_layer in self.fc_layers:
            x = self.dropout_layer(x)
            x = fc_layer(x)

        # Within period prediction.
        within_period_x = self.input_projection2(within_period_x)
        within_period_x += self.pos_encoding2
        for transformer_layer in self.transformer_layers2:
            within_period_x = transformer_layer(within_period_x)
        within_period_x = flatten_sequential_feats(within_period_x,
                                                batch_size,
                                                self.num_frames)

        for fc_layer in self.within_period_fc_layers:
            within_period_x = self.dropout_layer(within_period_x)
            within_period_x = fc_layer(within_period_x)

        return x,within_period_x,affinity


def get_repnet_model(logdir):
    """Returns a trained RepNet model.

    Args:
        logdir (string): Path to directory where checkpoint will be downloaded.

    Returns:
        model (Keras model): Trained RepNet model.
    """
    # Check if we are in eager mode.
    assert tf.executing_eagerly()

    # Models will be called in eval mode.
    tf.keras.backend.set_learning_phase(0)

    # Define RepNet model.
    model = NoFeatureExtractPeriodEstimator()
    # tf.function for speed.
    model.call = tf.function(model.call)

    # Define checkpoint and checkpoint manager.
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, directory=logdir, max_to_keep=10)
    latest_ckpt = ckpt_manager.latest_checkpoint
    print('Loading from: ', latest_ckpt)
    if not latest_ckpt:
        raise ValueError('Path does not have a checkpoint to load.')
    # Restore weights.
    ckpt.restore(latest_ckpt).expect_partial()

    # Pass dummy frames to build graph.

    model(tf.random.uniform((1,64, 64,1)))      
    # for v in model.variables:
    #     print(v.name)
    #     print('-'*50)
    #     print(v)
    
    return model

# def load_weight(model):
#     # ckpt = tf.train.Checkpoint(model=model)
#     for layer in model.layers:
#         print(layer.name)
    

