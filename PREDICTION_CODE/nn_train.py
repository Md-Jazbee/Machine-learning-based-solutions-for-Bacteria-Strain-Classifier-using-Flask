
import numpy as np
import tensorflow as tf
import re
import os
import sys
import errno
import warnings
from contextlib import redirect_stdout
import math

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, Lambda, Masking
from tensorflow.keras.layers import concatenate, add, multiply, average, maximum, Flatten
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import glorot_uniform, he_uniform, orthogonal
from tensorflow.keras.models import load_model

from utils import ReadSequence, CSVMemoryLogger, set_mem_growth, DatasetParser


class RCConfig:

    """
    RCNet configuration class.

    """

    def __init__(self, config):
        """RCConfig constructor"""
        try:
            self.strategy_dict = {
                "MirroredStrategy": tf.distribute.MirroredStrategy,
                "OneDeviceStrategy": tf.distribute.OneDeviceStrategy,
                "CentralStorageStrategy": tf.distribute.experimental.CentralStorageStrategy,
                "MultiWorkerMirroredStrategy": tf.distribute.experimental.MultiWorkerMirroredStrategy,
                "TPUStrategy": tf.distribute.experimental.TPUStrategy,
            }

            # Devices Config #
            # Get the number of available GPUs
            try:
                self.strategy = config['Devices']['DistStrategy']
            except KeyError:
                print("Unknown distribution strategy. Using MirroredStrategy.")
                self.strategy = "MirroredStrategy"
            self.__n_gpus = 0
            self.tpu_strategy = None

            # for using tf.device instead of strategy
            try:
                self.simple_build = config['Devices'].getboolean('SimpleBuild')
            except KeyError:
                self.simple_build = False
            self.base_batch_size = config['Training'].getint('BatchSize')
            self.batch_size = self.base_batch_size

            self.set_n_gpus()
            self.model_build_device = config['Devices']['Device_build']

            # Data Loading Config #
            # If using generators to load data batch by batch, set up the number of batch workers and the queue size
            self.use_generators_keras = config['DataLoad'].getboolean('LoadTrainingByBatch')
            self.use_tf_data = config['DataLoad'].getboolean('Use_TFData')
            if self.use_generators_keras:
                self.multiprocessing = config['DataLoad'].getboolean('Multiprocessing')
                self.batch_loading_workers = config['DataLoad'].getint('BatchWorkers')
                self.batch_queue = config['DataLoad'].getint('BatchQueue')

            # Input Data Config #
            # Set the sequence length and the alphabet
            self.seq_length = config['InputData'].getint('SeqLength')
            self.alphabet = "ACGT"
            self.seq_dim = len(self.alphabet)
            try:
                self.mask_zeros = config['InputData'].getboolean('MaskZeros')
            except KeyError:
                self.mask_zeros = False
            # subread settings (subread = first k nucleotides of a read)
            self.use_subreads = config['InputData'].getboolean('UseSubreads')
            self.min_subread_length = config['InputData'].getint('MinSubreadLength')
            self.max_subread_length = config['InputData'].getint('MaxSubreadLength')
            self.dist_subread = config['InputData']['DistSubread']

            # Architecture Config #
            # Set the seed
            self.seed = config['Architecture'].getint('Seed')
            # Set the initializer (choose between He and Glorot uniform)
            self.init_mode = config['Architecture']['WeightInit']
            if self.init_mode == 'he_uniform':
                self.initializer = he_uniform(self.seed)
            elif self.init_mode == 'glorot_uniform':
                self.initializer = glorot_uniform(self.seed)
            else:
                raise ValueError('Unknown initializer')
            self.ortho_gain = config['Architecture'].getfloat('OrthoGain')

            # Define the network architecture
            self.rc_mode = config['Architecture']['RC_Mode']
            self.n_conv = config['Architecture'].getint('N_Conv')
            self.n_recurrent = config['Architecture'].getint('N_Recurrent')
            self.n_dense = config['Architecture'].getint('N_Dense')
            self.input_dropout = config['Architecture'].getfloat('Input_Dropout')
            self.conv_units = [int(u) for u in config['Architecture']['Conv_Units'].split(',')]
            self.conv_filter_size = [int(s) for s in config['Architecture']['Conv_FilterSize'].split(',')]
            self.conv_activation = config['Architecture']['Conv_Activation']
            self.conv_bn = config['Architecture'].getboolean('Conv_BN')
            self.conv_pooling = config['Architecture']['Conv_Pooling']
            self.conv_dropout = config['Architecture'].getfloat('Conv_Dropout')
            self.recurrent_units = [int(u) for u in config['Architecture']['Recurrent_Units'].split(',')]
            self.recurrent_bn = config['Architecture'].getboolean('Recurrent_BN')
            if self.n_recurrent == 1 and self.recurrent_bn:
                raise ValueError("RC-BN is intended for RC layers with 2D output. Use RC-Conv1D or RC-LSTM returning"
                                 " sequences.")
            self.recurrent_dropout = config['Architecture'].getfloat('Recurrent_Dropout')
            merge_dict = {
                # motif on fwd fuzzy OR rc (Goedel t-conorm)
                "maximum": maximum,
                # motif on fwd fuzzy AND rc (product t-norm)
                "multiply": multiply,
                # motif on fwd PLUS/"OR" rc (Shrikumar-style)
                "add": add,
                # motif on fwd PLUS/"OR" rc (Shrikumar-style), rescaled
                "average": average
            }
            if self.rc_mode != "none":
                self.dense_merge = merge_dict.get(config['Architecture']['Dense_Merge'])
                if self.dense_merge is None:
                    raise ValueError('Unknown dense merge function')
            self.dense_units = [int(u) for u in config['Architecture']['Dense_Units'].split(',')]
            self.dense_activation = config['Architecture']['Dense_Activation']
            self.dense_bn = config['Architecture'].getboolean('Dense_BN')
            self.dense_dropout = config['Architecture'].getfloat('Dense_Dropout')

            # If needed, weight classes
            self.use_weights = config['ClassWeights'].getboolean('UseWeights')
            if self.use_weights:
                try:
                    counts = [float(x) for x in config['ClassWeights']['ClassCounts'].split(',')]
                except KeyError:
                    counts = [config['ClassWeights'].getfloat('ClassCount_0'),
                              config['ClassWeights'].getfloat('ClassCount_1')]
                sum_count = sum(counts)
                weights = [sum_count/(2*class_count) for class_count in counts]
                classes = range(len(counts))
                self.class_weight = dict(zip(classes, weights))
                self.log_init = False
                if self.log_init:
                    self.output_bias = tf.keras.initializers.Constant(np.log(counts[1]/counts[0]))
                else:
                    self.output_bias = 'zeros'
            else:
                self.class_weight = None
                self.output_bias = 'zeros'

            # Paths Config #
            # Set the input data paths
            self.x_train_path = config['Paths']['TrainingData']
            self.y_train_path = config['Paths']['TrainingLabels']
            self.x_val_path = config['Paths']['ValidationData']
            self.y_val_path = config['Paths']['ValidationLabels']
            # Set the run name
            self.runname = config['Paths']['RunName']

            # Training Config #
            # Set the number op epochs, batch size and the optimizer
            self.epoch_start = config['Training'].getint('EpochStart') - 1
            self.epoch_end = config['Training'].getint('EpochEnd') - 1

            self.patience = config['Training'].getint('Patience')
            self.l2 = config['Training'].getfloat('Lambda_L2')
            self.regularizer = regularizers.l2(self.l2)
            self.learning_rate = config['Training'].getfloat('LearningRate')
            self.optimization_method = config['Training']['Optimizer']
            if self.optimization_method == "adam":
                self.optimizer = Adam(lr=self.learning_rate)
            else:
                warnings.warn("Custom learning rates implemented for Adam only. Using default Keras learning rate.")
                self.optimizer = self.optimization_method
            # If needed, log the memory usage
            self.log_memory = config['Training'].getboolean('MemUsageLog')
            self.summaries = config['Training'].getboolean('Summaries')
            self.log_superpath = config['Training']['LogPath']
            self.log_dir = os.path.join(self.log_superpath, "{runname}-logs".format(runname=self.runname))

            self.use_tb = config['Training'].getboolean('Use_TB')
            if self.use_tb:
                self.tb_hist_freq = config['Training'].getint('TBHistFreq')
        except KeyError as ke:
            sys.exit("The config file is not compatible with this version "
                     "Missing keyword: {}".format(ke))
        except AttributeError as ae:
            sys.exit("The config file is not compatible with this version. "
                     "Error: {}".format(ae))

    def set_tf_session(self):
        """Set TF session."""
        # If no GPUs, use CPUs
        if self.__n_gpus == 0:
            self.model_build_device = '/cpu:0'
        set_mem_growth()

    def set_n_gpus(self):
        self.__n_gpus = len(tf.config.get_visible_devices('GPU'))
        self.batch_size = self.base_batch_size * self.__n_gpus if self.__n_gpus > 0 else self.base_batch_size

    def get_n_gpus(self):
        return self.__n_gpus

    def set_tpu_resolver(self, tpu_resolver):
        if tpu_resolver is not None:
            self.tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu_resolver)
            self.batch_size = self.base_batch_size * self.tpu_strategy.num_replicas_in_sync


class RCNet:

    """
    Reverse-complement neural network class.

    """

    def __init__(self, config, training_mode=True, verbose_load=False):
        """RCNet constructor and config parsing"""
        self.config = config
        if self.config.use_tf_data and not tf.executing_eagerly():
            warnings.warn("Training with TFRecordDatasets supported only in eager mode. Looking for .npy files...")
            self.config.use_tf_data = False

        self.config.set_tf_session()
        self.history = None
        self.verbose_load = verbose_load

        self.__t_sequence = None
        self.__v_sequence = None
        self.training_sequence = None
        self.x_train = None
        self.y_train = None
        self.length_train = 0
        self.val_indices = None
        self.x_val = None
        self.y_val = None
        self.validation_data = (self.x_val, self.y_val)
        self.length_val = 0
        self.model = None

        if training_mode:
            try:
                os.makedirs(self.config.log_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            self.__set_callbacks()

        if self.config.epoch_start > 0:
            checkpoint_name = self.config.log_dir + "/nn-{runname}-".format(runname=self.config.runname)
            self.model = load_model(checkpoint_name + "e{epoch:03d}.h5".format(epoch=self.config.epoch_start-1))
        else:
            # Build the model using the CPU or GPU or TPU
            if self.config.tpu_strategy is not None:
                self.strategy = self.config.tpu_strategy
            elif self.config.simple_build:
                self.strategy = None
            elif self.config.strategy == "OneDeviceStrategy":
                self.strategy = self.config.strategy_dict[self.config.strategy](self.config.model_build_device)
            else:
                self.strategy = self.config.strategy_dict[self.config.strategy]()

            with self.get_device_strategy_scope():
                if self.config.rc_mode == "full":
                    self.__build_rc_model()
                elif self.config.rc_mode == "siam":
                    self.__build_siam_model()
                elif self.config.rc_mode == "none":
                    self.__build_simple_model()
                else:
                    raise ValueError('Unrecognized RC mode')

    def get_device_strategy_scope(self):
        if self.config.simple_build:
            device_strategy_scope = tf.device(self.config.model_build_device)
        else:
            device_strategy_scope = self.strategy.scope()
        return device_strategy_scope

    def load_data(self):
        """Load datasets"""
        print("Loading...")

        if self.config.use_tf_data:
            prefetch_size = tf.data.experimental.AUTOTUNE

            def count_data_items(filenames):
                n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
                return np.max(n) + 1

            parser = DatasetParser(self.config.seq_length)
            train_filenames = tf.io.gfile.glob(self.config.x_train_path + "/*.tfrec")
            self.length_train = count_data_items(train_filenames)
            self.training_sequence = \
                parser.read_dataset(train_filenames).shuffle(buffer_size=self.config.batch_size*self.config.batch_queue)
            self.training_sequence = \
                self.training_sequence.repeat().batch(self.config.batch_size).prefetch(prefetch_size)

            val_filenames = tf.io.gfile.glob(self.config.x_val_path + "/*.tfrec")
            self.length_val = count_data_items(val_filenames)
            self.validation_data = \
                parser.read_dataset(val_filenames).repeat().batch(self.config.batch_size).prefetch(prefetch_size)
        elif self.config.use_generators_keras:
            # Prepare the generators for loading data batch by batch
            self.x_train = np.load(self.config.x_train_path, mmap_mode='r')
            self.y_train = np.load(self.config.y_train_path, mmap_mode='r')
            self.__t_sequence = ReadSequence(self.x_train, self.y_train, self.config.batch_size,
                                             self.config.use_subreads, self.config.min_subread_length,
                                             self.config.max_subread_length, self.config.dist_subread,
                                             verbose_id="TRAIN" if self.verbose_load else None)

            self.training_sequence = self.__t_sequence
            self.length_train = len(self.x_train)

            # Prepare the generators for loading data batch by batch
            self.x_val = np.load(self.config.x_val_path, mmap_mode='r')
            self.y_val = np.load(self.config.y_val_path, mmap_mode='r')
            self.__v_sequence = ReadSequence(self.x_val, self.y_val, self.config.batch_size,
                                             self.config.use_subreads, self.config.min_subread_length,
                                             self.config.max_subread_length, self.config.dist_subread,
                                             verbose_id="VAL" if self.verbose_load else None)
            self.validation_data = self.__v_sequence

            self.length_val = len(self.x_val)
        else:
            # ... or load all the data to memory
            self.x_train = np.load(self.config.x_train_path)
            self.y_train = np.load(self.config.y_train_path)
            self.length_train = self.x_train.shape

            # ... or load all the data to memory
            self.x_val = np.load(self.config.x_val_path)
            self.y_val = np.load(self.config.y_val_path)
            self.val_indices = np.arange(len(self.y_val))
            np.random.shuffle(self.val_indices)
            self.x_val = self.x_val[self.val_indices]
            self.y_val = self.y_val[self.val_indices]
            self.validation_data = (self.x_val, self.y_val)
            self.length_val = self.x_val.shape[0]

    def __add_lstm(self, inputs, return_sequences):
        # LSTM with sigmoid activation corresponds to the CuDNNLSTM
        if not tf.executing_eagerly() and self.config.get_n_gpus() > 0:
            x = Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(self.config.recurrent_units[0],
                                                                  kernel_initializer=self.config.initializer,
                                                                  recurrent_initializer=orthogonal(
                                                                      gain=self.config.ortho_gain,
                                                                      seed=self.config.seed),
                                                                  kernel_regularizer=self.config.regularizer,
                                                                  return_sequences=return_sequences))(inputs)
        else:
            x = Bidirectional(LSTM(self.config.recurrent_units[0], kernel_initializer=self.config.initializer,
                                   recurrent_initializer=orthogonal(gain=self.config.ortho_gain,
                                                                    seed=self.config.seed),
                                   kernel_regularizer=self.config.regularizer,
                                   return_sequences=return_sequences,
                                   recurrent_activation='sigmoid'))(inputs)
        return x

    def __add_siam_lstm(self, inputs_fwd, inputs_rc, return_sequences, units):
        # LSTM with sigmoid activation corresponds to the CuDNNLSTM
        if not tf.executing_eagerly() and self.config.get_n_gpus() > 0:
            shared_lstm = Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(units,
                                                                            kernel_initializer=self.config.initializer,
                                                                            recurrent_initializer=orthogonal(
                                                                                gain=self.config.ortho_gain,
                                                                                seed=self.config.seed),
                                                                            kernel_regularizer=self.config.regularizer,
                                                                            return_sequences=return_sequences))
        else:
            shared_lstm = Bidirectional(LSTM(units, kernel_initializer=self.config.initializer,
                                             recurrent_initializer=orthogonal(gain=self.config.ortho_gain,
                                                                              seed=self.config.seed),
                                             kernel_regularizer=self.config.regularizer,
                                             return_sequences=return_sequences,
                                             recurrent_activation='sigmoid'))

        x_fwd = shared_lstm(inputs_fwd)
        x_rc = shared_lstm(inputs_rc)
        if return_sequences:
            rev_axes = (1, 2)
        else:
            rev_axes = 1
        revcomp_out = Lambda(lambda x: K.reverse(x, axes=rev_axes), output_shape=shared_lstm.output_shape[1:],
                             name="reverse_lstm_output_{n}".format(n=self.__current_recurrent+1))
        x_rc = revcomp_out(x_rc)
        return x_fwd, x_rc

    def __add_rc_lstm(self, inputs, return_sequences, units):
        revcomp_in = Lambda(lambda x: K.reverse(x, axes=(1, 2)), output_shape=inputs.shape[1:],
                            name="reverse_complement_lstm_input_{n}".format(n=self.__current_recurrent+1))
        inputs_rc = revcomp_in(inputs)
        x_fwd, x_rc = self.__add_siam_lstm(inputs, inputs_rc, return_sequences, units)
        out = concatenate([x_fwd, x_rc], axis=-1)
        return out

    def __add_siam_conv1d(self, inputs_fwd, inputs_rc, units):
        shared_conv = Conv1D(units, self.config.conv_filter_size[0], padding='same',
                             kernel_regularizer=self.config.regularizer)
        x_fwd = shared_conv(inputs_fwd)
        x_rc = shared_conv(inputs_rc)
        revcomp_out = Lambda(lambda x: K.reverse(x, axes=(1, 2)), output_shape=shared_conv.output_shape[1:],
                             name="reverse_complement_conv1d_output_{n}".format(n=self.__current_conv+1))
        x_rc = revcomp_out(x_rc)
        return x_fwd, x_rc

    def __add_rc_conv1d(self, inputs, units):
        revcomp_in = Lambda(lambda x: K.reverse(x, axes=(1, 2)), output_shape=inputs.shape[1:],
                            name="reverse_complement_conv1d_input_{n}".format(n=self.__current_conv+1))
        inputs_rc = revcomp_in(inputs)
        x_fwd, x_rc = self.__add_siam_conv1d(inputs, inputs_rc, units)
        out = concatenate([x_fwd, x_rc], axis=-1)
        return out

    def __add_siam_batchnorm(self, inputs_fwd, inputs_rc):
        input_shape = inputs_rc.shape
        if len(input_shape) != 3:
            raise ValueError("Intended for RC layers with 2D output. Use RC-Conv1D or RC-LSTM returning sequences."
                             "Expected dimension: 3, but got: " + str(len(input_shape)))
        rc_in = Lambda(lambda x: K.reverse(x, axes=(1, 2)), output_shape=input_shape[1:],
                       name="reverse_complement_batchnorm_input_{n}".format(n=self.__current_bn+1))
        inputs_rc = rc_in(inputs_rc)
        out = concatenate([inputs_fwd, inputs_rc], axis=1)
        out = BatchNormalization()(out)
        split_shape = out.shape[1] // 2
        new_shape = [split_shape, input_shape[2]]
        fwd_out = Lambda(lambda x: x[:, :split_shape, :], output_shape=new_shape,
                         name="split_batchnorm_fwd_output_{n}".format(n=self.__current_bn+1))
        rc_out = Lambda(lambda x: K.reverse(x[:, split_shape:, :], axes=(1, 2)), output_shape=new_shape,
                        name="split_batchnorm_rc_output_{n}".format(n=self.__current_bn+1))

        x_fwd = fwd_out(out)
        x_rc = rc_out(out)
        return x_fwd, x_rc

    def __add_rc_batchnorm(self, inputs):
        input_shape = inputs.shape
        if len(input_shape) != 3:
            raise ValueError("Intended for RC layers with 2D output. Use RC-Conv1D or RC-LSTM returning sequences."
                             "Expected dimension: 3, but got: " + str(len(input_shape)))
        split_shape = inputs.shape[-1] // 2
        new_shape = [input_shape[1], split_shape]
        fwd_in = Lambda(lambda x: x[:, :, :split_shape], output_shape=new_shape,
                        name="split_batchnorm_fwd_input_{n}".format(n=self.__current_bn+1))
        rc_in = Lambda(lambda x: x[:, :, split_shape:], output_shape=new_shape,
                       name="split_batchnorm_rc_input_{n}".format(n=self.__current_bn+1))
        inputs_fwd = fwd_in(inputs)
        inputs_rc = rc_in(inputs)
        x_fwd, x_rc = self.__add_siam_batchnorm(inputs_fwd, inputs_rc)
        out = concatenate([x_fwd, x_rc], axis=-1)
        return out

    def __add_siam_merge_dense(self, inputs_fwd, inputs_rc, units, merge_function=add):
        shared_dense = Dense(units, kernel_regularizer=self.config.regularizer)
        rc_in = Lambda(lambda x: K.reverse(x, axes=1), output_shape=inputs_rc.shape[1:],
                       name="reverse_merging_dense_input_{n}".format(n=1))
        inputs_rc = rc_in(inputs_rc)
        x_fwd = shared_dense(inputs_fwd)
        x_rc = shared_dense(inputs_rc)
        out = merge_function([x_fwd, x_rc])
        return out

    def __add_rc_merge_dense(self, inputs, units, merge_function=add):
        split_shape = inputs.shape[-1] // 2
        fwd_in = Lambda(lambda x: x[:, :split_shape], output_shape=[split_shape],
                        name="split_merging_dense_input_fwd_{n}".format(n=1))
        rc_in = Lambda(lambda x: x[:, split_shape:], output_shape=[split_shape],
                       name="split_merging_dense_input_rc_{n}".format(n=1))
        x_fwd = fwd_in(inputs)
        x_rc = rc_in(inputs)
        return self.__add_siam_merge_dense(x_fwd, x_rc, units, merge_function)

    def __build_simple_model(self):
        """Build the standard network"""
        print("Building model...")
        # Number of added recurrent layers
        self.__current_recurrent = 0
        # Initialize input
        inputs = Input(shape=(self.config.seq_length, self.config.seq_dim))
        if self.config.mask_zeros:
            x = Masking()(inputs)
        else:
            x = inputs
        # The last recurrent layer should return the output for the last unit only.
        # Previous layers must return output for all units
        return_sequences = True if self.config.n_recurrent > 1 else False
        # Input dropout
        if not np.isclose(self.config.input_dropout, 0.0):
            x = Dropout(self.config.input_dropout, seed=self.config.seed)(x)
        else:
            x = inputs
        # First convolutional/recurrent layer
        if self.config.n_conv > 0:
            # Convolutional layers will always be placed before recurrent ones
            # Standard convolutional layer
            x = Conv1D(self.config.conv_units[0], self.config.conv_filter_size[0], padding='same',
                       kernel_regularizer=self.config.regularizer)(x)
            if self.config.conv_bn:
                # Standard batch normalization layer
                x = BatchNormalization()(x)
            # Add activation
            x = Activation(self.config.conv_activation)(x)
        elif self.config.n_recurrent > 0:
            # If no convolutional layers, the first layer is recurrent.
            # CuDNNLSTM requires a GPU and tensorflow with cuDNN
            x = self.__add_lstm(x, return_sequences)
            if self.config.recurrent_bn and return_sequences:
                # Standard batch normalization layer
                x = BatchNormalization()(x)
            # Add dropout
            x = Dropout(self.config.recurrent_dropout, seed=self.config.seed)(x)
            # First recurrent layer already added
            self.__current_recurrent = 1
        else:
            raise ValueError('First layer should be convolutional or recurrent')

        # For next convolutional layers
        for i in range(1, self.config.n_conv):
            # Add pooling first
            if self.config.conv_pooling == 'max':
                x = MaxPooling1D()(x)
            elif self.config.conv_pooling == 'average':
                x = AveragePooling1D()(x)
            elif not (self.config.conv_pooling in ['last_max', 'last_average', 'none']):
                # Skip pooling if it should be applied to the last conv layer or skipped altogether.
                # Throw a ValueError if the pooling method is unrecognized.
                raise ValueError('Unknown pooling method')
            # Add dropout (drops whole features)
            if not np.isclose(self.config.conv_dropout, 0.0):
                x = Dropout(self.config.conv_dropout, seed=self.config.seed)(x)
            # Add layer
            # Standard convolutional layer
            x = Conv1D(self.config.conv_units[i], self.config.conv_filter_size[i], padding='same',
                       kernel_initializer=self.config.initializer, kernel_regularizer=self.config.regularizer)(x)
            # Add batch norm
            if self.config.conv_bn:
                # Standard batch normalization layer
                x = BatchNormalization()(x)
            # Add activation
            x = Activation(self.config.conv_activation)(x)

        # Pooling layer
        if self.config.n_conv > 0:
            if self.config.conv_pooling == 'max' or self.config.conv_pooling == 'last_max':
                if self.config.n_recurrent == 0:
                    # If no recurrent layers, use global pooling
                    x = GlobalMaxPooling1D()(x)
                else:
                    # for recurrent layers, use normal pooling
                    x = MaxPooling1D()(x)
            elif self.config.conv_pooling == 'average' or self.config.conv_pooling == 'last_average':
                if self.config.n_recurrent == 0:
                    # if no recurrent layers, use global pooling
                    x = GlobalAveragePooling1D()(x)
                else:
                    # for recurrent layers, use normal pooling
                    x = AveragePooling1D()(x)
            elif self.config.conv_pooling == 'none':
                x = Flatten()(x)
            else:
                # Skip pooling if needed or throw a ValueError if the pooling method is unrecognized
                # (should be thrown above)
                raise ValueError('Unknown pooling method')
            # Add dropout (drops whole features)
            if not np.isclose(self.config.conv_dropout, 0.0):
                x = Dropout(self.config.conv_dropout, seed=self.config.seed)(x)

        # Recurrent layers
        for i in range(self.__current_recurrent, self.config.n_recurrent):
            if i == self.config.n_recurrent - 1:
                # In the last layer, return output only for the last unit
                return_sequences = False
            # Add a bidirectional recurrent layer. CuDNNLSTM requires a GPU and tensorflow with cuDNN
            x = self.__add_lstm(inputs, return_sequences)
            if self.config.recurrent_bn and return_sequences:
                # Standard batch normalization layer
                x = BatchNormalization()(x)
            # Add dropout
            x = Dropout(self.config.recurrent_dropout, seed=self.config.seed)(x)

        # Dense layers
        for i in range(0, self.config.n_dense):
            x = Dense(self.config.dense_units[i], kernel_regularizer=self.config.regularizer)(x)
            if self.config.dense_bn:
                # Standard batch normalization layer
                x = BatchNormalization()(x)
            x = Activation(self.config.dense_activation)(x)
            x = Dropout(self.config.dense_dropout, seed=self.config.seed)(x)

        # Output layer for binary classification
        x = Dense(1, kernel_regularizer=self.config.regularizer, bias_initializer=self.config.output_bias)(x)
        x = Activation('sigmoid')(x)

        # Initialize the model
        self.model = Model(inputs, x)

    def __build_rc_model(self):
        """Build the RC network"""
        print("Building RC-model...")
        # Number of added recurrent layers
        self.__current_recurrent = 0
        self.__current_conv = 0
        self.__current_bn = 0
        # Initialize input
        inputs = Input(shape=(self.config.seq_length, self.config.seq_dim))
        if self.config.mask_zeros:
            x = Masking()(inputs)
        else:
            x = inputs
        # The last recurrent layer should return the output for the last unit only.
        #  Previous layers must return output for all units
        return_sequences = True if self.config.n_recurrent > 1 else False
        # Input dropout
        if not np.isclose(self.config.input_dropout, 0.0):
            x = Dropout(self.config.input_dropout, seed=self.config.seed)(x)
        else:
            x = inputs
        # First convolutional/recurrent layer
        if self.config.n_conv > 0:
            # Convolutional layers will always be placed before recurrent ones
            x = self.__add_rc_conv1d(x, self.config.conv_units[0])
            if self.config.conv_bn:
                # Reverse-complemented batch normalization layer
                x = self.__add_rc_batchnorm(x)
                self.__current_bn = self.__current_bn + 1
            x = Activation(self.config.conv_activation)(x)
            self.__current_conv = self.__current_conv + 1
        elif self.config.n_recurrent > 0:
            # If no convolutional layers, the first layer is recurrent.
            # CuDNNLSTM requires a GPU and tensorflow with cuDNN
            x = self.__add_rc_lstm(x, return_sequences, self.config.recurrent_units[0])
            if self.config.recurrent_bn and return_sequences:
                # Reverse-complemented batch normalization layer
                x = self.__add_rc_batchnorm(x)
                self.__current_bn = self.__current_bn + 1
            # Add dropout
            if not np.isclose(self.config.recurrent_dropout, 0.0):
                x = Dropout(self.config.recurrent_dropout, seed=self.config.seed)(x)
            # First recurrent layer already added
            self.__current_recurrent = self.__current_recurrent + 1
        else:
            raise ValueError('First layer should be convolutional or recurrent')

        # For next convolutional layers
        for i in range(1, self.config.n_conv):
            # Add pooling first
            if self.config.conv_pooling == 'max':
                x = MaxPooling1D()(x)
            elif self.config.conv_pooling == 'average':
                x = AveragePooling1D()(x)
            elif not (self.config.conv_pooling in ['last_max', 'last_average', 'none']):
                # Skip pooling if it should be applied to the last conv layer or skipped altogether.
                # Throw a ValueError if the pooling method is unrecognized.
                raise ValueError('Unknown pooling method')
            # Add dropout (drops whole features)
            if not np.isclose(self.config.conv_dropout, 0.0):
                x = Dropout(self.config.conv_dropout, seed=self.config.seed)(x)
            # Add layer
            x = self.__add_rc_conv1d(x, self.config.conv_units[i])
            if self.config.conv_bn:
                # Reverse-complemented batch normalization layer
                x = self.__add_rc_batchnorm(x)
                self.__current_bn = self.__current_bn + 1
            x = Activation(self.config.conv_activation)(x)
            self.__current_conv = self.__current_conv + 1

        # Pooling layer
        if self.config.n_conv > 0:
            if self.config.conv_pooling == 'max' or self.config.conv_pooling == 'last_max':
                if self.config.n_recurrent == 0:
                    # If no recurrent layers, use global pooling
                    x = GlobalMaxPooling1D()(x)
                else:
                    # for recurrent layers, use normal pooling
                    x = MaxPooling1D()(x)
            elif self.config.conv_pooling == 'average' or self.config.conv_pooling == 'last_average':
                if self.config.n_recurrent == 0:
                    # if no recurrent layers, use global pooling
                    x = GlobalAveragePooling1D()(x)
                else:
                    # for recurrent layers, use normal pooling
                    x = AveragePooling1D()(x)
            elif self.config.conv_pooling == 'none':
                x = Flatten()(x)
            else:
                # Skip pooling if needed or throw a ValueError if the pooling method is unrecognized
                # (should be thrown above)
                raise ValueError('Unknown pooling method')
            # Add dropout (drops whole features)
            if not np.isclose(self.config.conv_dropout, 0.0):
                x = Dropout(self.config.conv_dropout, seed=self.config.seed)(x)

        # Recurrent layers
        for i in range(self.__current_recurrent, self.config.n_recurrent):
            if i == self.config.n_recurrent - 1:
                # In the last layer, return output only for the last unit
                return_sequences = False
            # Add a bidirectional recurrent layer. CuDNNLSTM requires a GPU and tensorflow with cuDNN
            x = self.__add_rc_lstm(x, return_sequences, self.config.recurrent_units[i])
            if self.config.recurrent_bn and return_sequences:
                # Reverse-complemented batch normalization layer
                x = self.__add_rc_batchnorm(x)
                self.__current_bn = self.__current_bn + 1
            # Add dropout
            if not np.isclose(self.config.recurrent_dropout, 0.0):
                x = Dropout(self.config.recurrent_dropout, seed=self.config.seed)(x)
            self.__current_recurrent = self.__current_recurrent + 1

        # Dense layers
        for i in range(0, self.config.n_dense):
            if i == 0:
                x = self.__add_rc_merge_dense(x, self.config.dense_units[i])
            else:
                x = Dense(self.config.dense_units[i],  kernel_regularizer=self.config.regularizer)(x)
            if self.config.dense_bn:
                x = BatchNormalization()(x)
            x = Activation(self.config.dense_activation)(x)
            if not np.isclose(self.config.dense_dropout, 0.0):
                x = Dropout(self.config.dense_dropout, seed=self.config.seed)(x)

        # Output layer for binary classification
        if self.config.n_dense == 0:
            x = self.__add_rc_merge_dense(x, 1)
        else:
            x = Dense(1, kernel_regularizer=self.config.regularizer, bias_initializer=self.config.output_bias)(x)
        x = Activation('sigmoid')(x)

        # Initialize the model
        self.model = Model(inputs, x)

    def __build_siam_model(self):
        """Build the RC network"""
        print("Building siamese RC-model...")
        # Number of added recurrent layers
        self.__current_recurrent = 0
        self.__current_conv = 0
        self.__current_bn = 0
        # Initialize input
        inputs_fwd = Input(shape=(self.config.seq_length, self.config.seq_dim))
        if self.config.mask_zeros:
            x_fwd = Masking()(inputs_fwd)
        else:
            x_fwd = inputs_fwd
        revcomp_in = Lambda(lambda _x: K.reverse(_x, axes=(1, 2)), output_shape=inputs_fwd.shape[1:],
                            name="reverse_complement_input_{n}".format(n=self.__current_recurrent+1))
        inputs_rc = revcomp_in(x_fwd)
        # The last recurrent layer should return the output for the last unit only.
        # Previous layers must return output for all units
        return_sequences = True if self.config.n_recurrent > 1 else False
        # Input dropout
        if not np.isclose(self.config.input_dropout, 0.0):
            x_fwd = Dropout(self.config.input_dropout, seed=self.config.seed)(x_fwd)
            x_rc = Dropout(self.config.input_dropout, seed=self.config.seed)(inputs_rc)
        else:
            x_fwd = inputs_fwd
            x_rc = inputs_rc
        # First convolutional/recurrent layer
        if self.config.n_conv > 0:
            # Convolutional layers will always be placed before recurrent ones
            # Reverse-complement convolutional layer
            x_fwd, x_rc = self.__add_siam_conv1d(x_fwd, x_rc, self.config.conv_units[0])
            if self.config.conv_bn:
                # Reverse-complemented batch normalization layer
                x_fwd, x_rc = self.__add_siam_batchnorm(x_fwd, x_rc)
                self.__current_bn = self.__current_bn + 1
            # Add activation
            x_fwd = Activation(self.config.conv_activation)(x_fwd)
            x_rc = Activation(self.config.conv_activation)(x_rc)
            self.__current_conv = self.__current_conv + 1
        elif self.config.n_recurrent > 0:
            # If no convolutional layers, the first layer is recurrent.
            # CuDNNLSTM requires a GPU and tensorflow with cuDNN
            # RevComp input
            x_fwd, x_rc = self.__add_siam_lstm(x_fwd, x_rc, return_sequences, self.config.recurrent_units[0])
            # Add batch norm
            if self.config.recurrent_bn and return_sequences:
                # reverse-complemented batch normalization layer
                x_fwd, x_rc = self.__add_siam_batchnorm(x_fwd, x_rc)
                self.__current_bn = self.__current_bn + 1
            # Add dropout
            if not np.isclose(self.config.recurrent_dropout, 0.0):
                x_fwd = Dropout(self.config.recurrent_dropout, seed=self.config.seed)(x_fwd)
                x_rc = Dropout(self.config.recurrent_dropout, seed=self.config.seed)(x_rc)
            # First recurrent layer already added
            self.__current_recurrent = 1
        else:
            raise ValueError('First layer should be convolutional or recurrent')

        # For next convolutional layers
        for i in range(1, self.config.n_conv):
            # Add pooling first
            if self.config.conv_pooling == 'max':
                x_fwd = MaxPooling1D()(x_fwd)
                x_rc = MaxPooling1D()(x_rc)
            elif self.config.conv_pooling == 'average':
                x_fwd = AveragePooling1D()(x_fwd)
                x_rc = AveragePooling1D()(x_rc)
            elif not (self.config.conv_pooling in ['last_max', 'last_average', 'none']):
                # Skip pooling if it should be applied to the last conv layer or skipped altogether.
                # Throw a ValueError if the pooling method is unrecognized.
                raise ValueError('Unknown pooling method')
            # Add dropout (drops whole features)
            if not np.isclose(self.config.conv_dropout, 0.0):
                x_fwd = Dropout(self.config.conv_dropout, seed=self.config.seed)(x_fwd)
                x_rc = Dropout(self.config.conv_dropout, seed=self.config.seed)(x_rc)
            # Add layer
            # Reverse-complement convolutional layer
            x_fwd, x_rc = self.__add_siam_conv1d(x_fwd, x_rc, self.config.conv_units[i])
            if self.config.conv_bn:
                # Reverse-complemented batch normalization layer
                x_fwd, x_rc = self.__add_siam_batchnorm(x_fwd, x_rc)
                self.__current_bn = self.__current_bn + 1
            # Add activation
            x_fwd = Activation(self.config.conv_activation)(x_fwd)
            x_rc = Activation(self.config.conv_activation)(x_rc)
            self.__current_conv = self.__current_conv + 1

        # Pooling layer
        if self.config.n_conv > 0:
            if self.config.conv_pooling == 'max' or self.config.conv_pooling == 'last_max':
                if self.config.n_recurrent == 0:
                    # If no recurrent layers, use global pooling
                    x_fwd = GlobalMaxPooling1D()(x_fwd)
                    x_rc = GlobalMaxPooling1D()(x_rc)
                else:
                    # for recurrent layers, use normal pooling
                    x_fwd = MaxPooling1D()(x_fwd)
                    x_rc = MaxPooling1D()(x_rc)
            elif self.config.conv_pooling == 'average' or self.config.conv_pooling == 'last_average':
                if self.config.n_recurrent == 0:
                    # if no recurrent layers, use global pooling
                    x_fwd = GlobalAveragePooling1D()(x_fwd)
                    x_rc = GlobalAveragePooling1D()(x_rc)
                else:
                    # for recurrent layers, use normal pooling
                    x_fwd = AveragePooling1D()(x_fwd)
                    x_rc = AveragePooling1D()(x_rc)
            elif self.config.conv_pooling != 'none':
                # Skip pooling if needed or throw a ValueError if the pooling method is unrecognized
                # (should be thrown above)
                raise ValueError('Unknown pooling method')
            # Add dropout (drops whole features)
            if not np.isclose(self.config.conv_dropout, 0.0):
                x_fwd = Dropout(self.config.conv_dropout, seed=self.config.seed)(x_fwd)
                x_rc = Dropout(self.config.conv_dropout, seed=self.config.seed)(x_rc)

        # Recurrent layers
        for i in range(self.__current_recurrent, self.config.n_recurrent):
            if i == self.config.n_recurrent - 1:
                # In the last layer, return output only for the last unit
                return_sequences = False
            # Add a bidirectional recurrent layer. CuDNNLSTM requires a GPU and tensorflow with cuDNN
            x_fwd, x_rc = self.__add_siam_lstm(x_fwd, x_rc, return_sequences, self.config.recurrent_units[i])
            # Add batch norm
            if self.config.recurrent_bn and return_sequences:
                # Reverse-complemented batch normalization layer
                x_fwd, x_rc = self.__add_siam_batchnorm(x_fwd, x_rc)
                self.__current_bn = self.__current_bn + 1
            # Add dropout
            if not np.isclose(self.config.recurrent_dropout, 0.0):
                x_fwd = Dropout(self.config.recurrent_dropout, seed=self.config.seed)(x_fwd)
                x_rc = Dropout(self.config.recurrent_dropout, seed=self.config.seed)(x_rc)

        # Output layer for binary classification
        if self.config.n_dense == 0:
            # Output layer for binary classification
            x = self.__add_siam_merge_dense(x_fwd, x_rc, 1)
        else:
            # Dense layers
            x = self.__add_siam_merge_dense(x_fwd, x_rc, self.config.dense_units[0])
            if self.config.dense_bn:
                # Batch normalization layer
                x = BatchNormalization()(x)
            x = Activation(self.config.dense_activation)(x)
            x = Dropout(self.config.dense_dropout, seed=self.config.seed)(x)
            for i in range(1, self.config.n_dense):
                x = Dense(self.config.dense_units[i], kernel_regularizer=self.config.regularizer)(x)
                if self.config.dense_bn:
                    # Batch normalization layer
                    x = BatchNormalization()(x)
                x = Activation(self.config.dense_activation)(x)
                x = Dropout(self.config.dense_dropout, seed=self.config.seed)(x)
            # Output layer for binary classification
            x = Dense(1, kernel_regularizer=self.config.regularizer, bias_initializer=self.config.output_bias)(x)
        x = Activation('sigmoid')(x)

        # Initialize the model
        self.model = Model(inputs_fwd, x)

    def compile_model(self):
        """Compile model and save model summaries"""
        if self.config.epoch_start == 0:
            print("Compiling...")
            self.model.compile(loss='binary_crossentropy',
                               optimizer=self.config.optimizer,
                               metrics=['accuracy'])

            # Print summary and plot model
            if self.config.summaries:
                with open(self.config.log_dir + "/summary-{runname}.txt".format(runname=self.config.runname), 'w') as f:
                    with redirect_stdout(f):
                        self.model.summary()
                plot_model(self.model,
                           to_file=self.config.log_dir + "/plot-{runname}.png".format(runname=self.config.runname),
                           show_shapes=False, rankdir='TB')
        else:
            print('Skipping compilation of a pre-trained model...')

    def __set_callbacks(self):
        """Set callbacks to use during training"""
        self.callbacks = []

        # Set early stopping
        self.callbacks.append(EarlyStopping(monitor="val_accuracy", patience=self.config.patience))

        # Add CSV callback with or without memory log
        if self.config.log_memory:
            self.callbacks.append(CSVMemoryLogger(
                self.config.log_dir + "/training-{runname}.csv".format(runname=self.config.runname),
                append=True))
        else:
            self.callbacks.append(CSVLogger(
                self.config.log_dir + "/training-{runname}.csv".format(runname=self.config.runname),
                append=True))
        # Save model after every epoch
        checkpoint_name = self.config.log_dir + "/nn-{runname}-".format(runname=self.config.runname)
        self.callbacks.append(ModelCheckpoint(filepath=checkpoint_name + "e{epoch:03d}.h5"))

        # Set TensorBoard
        if self.config.use_tb:
            self.callbacks.append(TensorBoard(
                log_dir=self.config.log_superpath + "/{runname}-tb".format(runname=self.config.runname),
                histogram_freq=self.config.tb_hist_freq, batch_size=self.config.batch_size,
                write_grads=True, write_images=True))

    def train(self):
        """Train the NN on Illumina reads using the supplied configuration."""
        print("Training...")
        with self.get_device_strategy_scope():
            if self.config.use_tf_data:
                # Fit a model using tf data
                self.history = self.model.fit(x=self.training_sequence,
                                              epochs=self.config.epoch_end,
                                              callbacks=self.callbacks,
                                              validation_data=self.validation_data,
                                              class_weight=self.config.class_weight,
                                              use_multiprocessing=self.config.multiprocessing,
                                              max_queue_size=self.config.batch_queue,
                                              workers=self.config.batch_loading_workers,
                                              initial_epoch=self.config.epoch_start,
                                              steps_per_epoch=math.ceil(self.length_train/self.config.batch_size),
                                              validation_steps=math.ceil(self.length_val/self.config.batch_size))

            elif self.config.use_generators_keras:
                # Fit a model using generators
                self.history = self.model.fit(x=self.training_sequence,
                                              epochs=self.config.epoch_end,
                                              callbacks=self.callbacks,
                                              validation_data=self.validation_data,
                                              class_weight=self.config.class_weight,
                                              use_multiprocessing=self.config.multiprocessing,
                                              max_queue_size=self.config.batch_queue,
                                              workers=self.config.batch_loading_workers,
                                              initial_epoch=self.config.epoch_start)
            else:
                # Fit a model using data in memory
                self.history = self.model.fit(x=self.x_train,
                                              y=self.y_train,
                                              batch_size=self.config.batch_size,
                                              epochs=self.config.epoch_end,
                                              callbacks=self.callbacks,
                                              validation_data=self.validation_data,
                                              shuffle=True,
                                              class_weight=self.config.class_weight,
                                              initial_epoch=self.config.epoch_start)
