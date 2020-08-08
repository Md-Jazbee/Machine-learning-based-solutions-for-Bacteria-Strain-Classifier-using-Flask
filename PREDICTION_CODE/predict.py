import time
import tensorflow as tf
import numpy as np
from Bio.SeqIO.FastaIO import SimpleFastaParser
import itertools
#from preproc import read_fasta, tokenize
from multiprocessing import Pool
from functools import partial
import gzip
import os
import math

def tokenize(seq, tokenizer, datatype='int8', read_length=250):
    matrix = tokenizer.texts_to_matrix(seq).astype(datatype)[:, 1:]
    if matrix.shape[0] < read_length:
        # Pad with zeros
        matrix = np.concatenate((matrix, np.zeros((read_length - len(seq), 4), dtype=datatype)))
    if matrix.shape[0] > read_length:
        # Trim
        matrix = matrix[:read_length, :]
    return matrix


def read_fasta(in_handle):
    for title, seq in SimpleFastaParser(in_handle):
        yield seq


def preproc(config):
    max_cores = config['Devices'].getint('N_CPUs')

    neg_path = config['InputPaths']['Fasta_Class_0']
    pos_path = config['InputPaths']['Fasta_Class_1']
    out_data_path = config['OutputPaths']['OutData']
    out_labels_path = config['OutputPaths']['OutLabels']

    do_shuffle = config['Options'].getboolean('Do_shuffle')
    if do_shuffle:
        seed = config['Options'].getint('ShuffleSeed')
        np.random.seed(seed)
    do_gzip = config['Options'].getboolean('Do_gzip')
    do_revc = config['Options'].getboolean('Do_revc')
    datatype = config['Options']['DataType']
    read_length = config['Options'].getint('ReadLength')
    use_tfdata = config['Options'].getboolean('Use_TFData')
    n_files = config['Options'].getint('N_Files')

    alphabet = "ACGT"
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(alphabet)
    # Preprocess
    if neg_path != "none":
        print("Preprocessing negative data...")
        with open(neg_path) as input_handle:
            if max_cores > 1:
                with Pool(processes=max_cores) as p:
                    x_train_neg = np.asarray(p.map(partial(tokenize, tokenizer=tokenizer, datatype=datatype,
                                                           read_length=read_length), read_fasta(input_handle)),
                                             dtype=datatype)
            else:
                x_train_neg = np.asarray(list(map(partial(tokenize, tokenizer=tokenizer, datatype=datatype,
                                                          read_length=read_length), read_fasta(input_handle))),
                                         dtype=datatype)
        n_negative = x_train_neg.shape[0]
    else:
        x_train_neg = np.zeros((0, read_length, 4),dtype=np.uint8)
        n_negative = 0

    if pos_path != "none":
        print("Preprocessing positive data...")
        with open(pos_path) as input_handle:
            # Parse fasta, tokenize in parallel & concatenate to negative data
            if max_cores > 1:
                with Pool(processes=max_cores) as p:
                    x_train_pos = np.asarray(p.map(partial(tokenize, tokenizer=tokenizer, datatype=datatype,
                                                           read_length=read_length), read_fasta(input_handle)),
                                             dtype=datatype)
            else:
                x_train_pos = np.asarray(list(map(partial(tokenize, tokenizer=tokenizer, datatype=datatype,
                                                          read_length=read_length), read_fasta(input_handle))),
                                         dtype=datatype)
        n_positive = x_train_pos.shape[0]
    else:
        x_train_pos = np.zeros((0, read_length, 4),dtype=np.uint8)
        n_positive = 0
    # Concatenate
    x_train = np.concatenate((x_train_neg, x_train_pos))
    # Add labels
    y_train = np.concatenate((np.repeat(0, n_negative).astype(datatype), np.repeat(1, n_positive).astype(datatype)))
    if do_revc:
        print("Augmenting data...")
        x_train = np.concatenate((x_train, x_train[::, ::-1, ::-1]))
        y_train = np.concatenate((y_train, y_train))

    if do_shuffle:
        indices = np.arange(len(y_train))
        np.random.shuffle(indices)
        x_train = x_train[indices, ::, ::]
        y_train = y_train[indices]

    # Save matrices #
    print("Saving data...")

    # Save output
    if not use_tfdata:
        # Compress output files
        if do_gzip:
            f_data = gzip.GzipFile(out_data_path + ".gz", "w")
            f_labels = gzip.GzipFile(out_labels_path + ".gz", "w")
        else:
            f_data = out_data_path
            f_labels = out_labels_path
        np.save(file=f_data, arr=x_train)
        np.save(file=f_labels, arr=y_train)
    else:
        out_dir = os.path.splitext(out_data_path)[0]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        n_all = n_negative + n_positive
        slice_size = math.ceil(n_all/n_files)
        for i in range(n_files):
            start = i * slice_size
            end = min((i+1) * slice_size, n_all)
            features_dataset = tf.data.Dataset.from_tensor_slices((x_train[start::end], y_train[start::end]))

            serialized_features_dataset = features_dataset.map(tf_serialize_example)

            filename = os.path.join(out_dir, os.path.splitext(os.path.basename(out_dir))[0]
                                    + '_{}-{}.tfrec'.format(start, end - 1))
            writer = tf.data.experimental.TFRecordWriter(filename)
            if tf.executing_eagerly():
                writer.write(serialized_features_dataset)
            else:
                with tf.compat.v1.Session() as sess:
                    sess.run(writer.write(serialized_features_dataset))

    print("Done!")


def predict_fasta(model, input_fasta, output, token_cores=8, datatype='int32'):

    alphabet = "ACGT"
    input_layer_id = [idx for idx, layer in enumerate(model.layers) if "Input" in str(layer)][0]
    read_length = model.get_layer(index=input_layer_id).get_output_at(0).shape[1]

    # Preproc
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(alphabet)

    print("Converting data to one_hot_encoding...")
    start = time.time()
    with open(input_fasta) as input_handle:
        with Pool(processes=token_cores) as p:
            x_data = np.asarray(p.map(partial(tokenize, tokenizer=tokenizer, datatype=datatype,
                                              read_length=read_length), read_fasta(input_handle)))
    # Predict
    print("Predicting the result")
    y_pred = np.ndarray.flatten(model.predict(x_data))
    end = time.time()
    print("Predictions for {} reads done in {} s".format(y_pred.shape[0], end - start))
    np.save(file=output, arr=y_pred)


def predict_npy(model, input_npy, output):
    x_data = np.load(input_npy)
    # Predict
    print("Predicting the result")
    start = time.time()
    y_pred = np.ndarray.flatten(model.predict(x_data))
    end = time.time()
    print("Predictions for {} reads done in {} s".format(y_pred.shape[0], end - start))
    np.save(file=output, arr=y_pred)
