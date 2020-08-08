
import re

import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model

from nn_train import RCConfig, RCNet


def convert_cudnn(config, saved_model, no_prep):

    path = saved_model
    if re.search("\.h5$", path) is not None:
        path = re.sub("\.h5$", "", path)

    if no_prep:
        weights_path = saved_model
    else:
        weights_path = path + "_weights.h5"
        # Prepare weights
        model = load_model(saved_model)
        model.save_weights(weights_path)

    # Load model architecture, device info and weights
    paprconfig = RCConfig(config)

    paprnet = RCNet(paprconfig, training_mode=False)

    paprnet.model.load_weights(weights_path)
    paprnet.model.compile(loss='binary_crossentropy',
                          optimizer=paprnet.config.optimizer,
                          metrics=['accuracy'])

    # Save output
    save_path = path + "_converted.h5"
    paprnet.model.save(save_path)
    print(paprnet.model.summary())
