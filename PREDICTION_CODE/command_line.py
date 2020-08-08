from predict import predict_fasta, predict_npy
from builtin_loading import BuiltinLoader
from utils import config_gpus, config_cpus, config_tpus
import sklearn
import tensorflow as tf
import random as rn
import argparse
import configparser
import os
import shutil
import multiprocessing
import numpy as np

def main():
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)
    rn.seed(seed)
    modulepath = os.path.dirname(__file__)
    builtin_configs = {"rapid": os.path.join(modulepath, "builtin", "config", "nn-img-rapid-cnn.ini"),
                       "sensitive": os.path.join(modulepath, "builtin", "config", "nn-img-sensitive-lstm.ini")}
    builtin_weights = {"rapid": os.path.join(modulepath, "builtin", "weights", "nn-img-rapid-cnn.h5"),
                       "sensitive": os.path.join(modulepath, "builtin", "weights", "nn-img-sensitive-lstm.h5")}
    runner = MainRunner(builtin_configs, builtin_weights)
    runner.parse()


def global_setup(args):
    tpu_resolver = None
    if args.tpu:
        tpu_resolver = config_tpus(args.tpu)
    if args.no_eager:
        print("Disabling eager mode...")
        tf.compat.v1.disable_eager_execution()
    if args.debug_device:
        tf.debugging.set_log_device_placement(True)
    if args.force_cpu:
        tf.config.set_visible_devices([], 'GPU')
        args.gpus = None
    default_verbosity = '3' if args.subparser == 'test' else '2'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.debug_tf) if args.debug_tf is not None else default_verbosity

    return tpu_resolver

def add_global_parser(gparser):
    gparser.add_argument('-v', '--version', dest='version', action='store_true', help='Print version.')
    gparser.add_argument('--debug-no-eager', dest="no_eager", help="Disable eager mode.",
                         default=False, action="store_true")
    gparser.add_argument('--debug-tf', dest="debug_tf", help="Set tensorflow debug info verbosity level. "
                                                             "0 = max, 3 = min. Default: 2 (errors);"
                                                             " 3 for tests (muted)", type=int)
    gparser.add_argument('--debug-device', dest="debug_device", help="Enable verbose device placement information.",
                         default=False, action="store_true")
    gparser.add_argument('--force-cpu', dest="force_cpu", help="Use a CPU even if GPUs are available.",
                         default=False, action="store_true")
    gparser.add_argument('--tpu', help="TPU name: 'colab' for Google Colab, or name of your TPU on GCE.")

    return gparser

class MainRunner:
    def __init__(self, builtin_configs=None, builtin_weights=None):
        self.builtin_configs = builtin_configs
        self.builtin_weights = builtin_weights
        self.bloader = BuiltinLoader(self.builtin_configs, self.builtin_weights)
        self.tpu_resolver = None
    
    def run_predict(self, args):
        if args.tpu is None:
            config_cpus(args.n_cpus)
            config_gpus(args.gpus)
        if args.output is None:
            args.output = os.path.splitext(args.input)[0] + "_predictions.npy"

        if args.sensitive:
            model = self.bloader.load_sensitive_model(training_mode=False, tpu_resolver=self.tpu_resolver)
        elif args.rapid:
            model = self.bloader.load_rapid_model(training_mode=False, tpu_resolver=self.tpu_resolver)
        else:
            if self.tpu_resolver is not None:
                tpu_strategy = tf.distribute.experimental.TPUStrategy(self.tpu_resolver)
                with tpu_strategy.scope():
                    model = tf.keras.models.load_model(args.custom)
            else:
                model = tf.keras.models.load_model(args.custom)

        if args.array:
            predict_npy(model, args.input, args.output)
        else:
            predict_fasta(model, args.input, args.output, args.n_cpus)

    def parse(self):
        parser = argparse.ArgumentParser(prog='deeac', description="neural networks.")
        parser = add_global_parser(parser)
        subparsers = parser.add_subparsers(help='subcommands. See command --help for details.', dest='subparser')

        # create the parser for the "predict" command
        parser_predict = subparsers.add_parser('predict', help='Predict for new data.')
        parser_predict.add_argument('input', help="Input file path [.fasta].")
        parser_predict.add_argument('-a', '--array', dest='array', action='store_true', help='Use .npy input instead.')
        predict_group = parser_predict.add_mutually_exclusive_group(required=True)
        predict_group.add_argument('-s', '--sensitive', dest='sensitive', action='store_true',
                                   help='Use the sensitive LSTM model.')
        predict_group.add_argument('-r', '--rapid', dest='rapid', action='store_true', help='Use the rapid CNN model.')
        predict_group.add_argument('-c', '--custom', dest='custom', help='Use the user-supplied, '
                                                                         'already compiled CUSTOM model.')
        parser_predict.add_argument('-o', '--output', help="Output file path [.npy].")
        parser_predict.add_argument('-n', '--n-cpus', dest="n_cpus", help="Number of CPU cores. Default: all.",
                                    type=int)
        parser_predict.add_argument('-g', '--gpus', dest="gpus", nargs='+', type=int,
                                    help="GPU devices to use (comma-separated). Default: all")
        parser_predict.set_defaults(func=self.run_predict)

        
        args = parser.parse_args()

        self.tpu_resolver = global_setup(args)

        if args.version:
            print("")
        elif hasattr(args, 'func'):
            args.func(args)
        else:
            print("")
            parser.print_help()


if __name__ == "__main__":
    main()
