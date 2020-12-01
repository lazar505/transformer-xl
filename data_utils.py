from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import json
import os
import pickle
import tensorflow as tf
from tensorflow.gfile import Exists as exists
from tensorflow.gfile import MakeDirs as makedirs

from vocabulary import Vocab


class Corpus(object):
    def __init__(self, path, dataset, *args, **kwargs):
        self.dataset = dataset
        self.vocab = Vocab(*args, **kwargs)

        self.vocab.count_file(os.path.join(path, "train.txt"))
        self.vocab.build_vocab()

        self.train = self.vocab.encode_file(os.path.join(path, "train.txt"), ordered=True)
        self.valid = self.vocab.encode_file(os.path.join(path, "valid.txt"), ordered=True)
        self.test = self.vocab.encode_file(os.path.join(path, "test.txt"), ordered=True)

        vocab_len = len(self.vocab)
        self.cutoffs = [0, int(vocab_len * 0.1), int(vocab_len * 0.2), int(vocab_len * 0.4)] + [vocab_len]
        # self.cutoffs = []

    def convert_to_tfrecords(self, split, save_dir, bsz, tgt_len, num_core_per_host, **kwargs):
        file_names = []

        record_name = "record_info-{}.bsz-{}.tlen-{}.json".format(split, bsz, tgt_len)

        record_info_path = os.path.join(save_dir, record_name)

        data = getattr(self, split)

        file_name, num_batch = create_ordered_tfrecords(save_dir, split, data, bsz, tgt_len)
        file_names.append(file_name)

        with open(record_info_path, "w") as fp:
            record_info = {
                "filenames": file_names,
                "num_batch": num_batch
            }
            json.dump(record_info, fp)


def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def batchify(data, batch_size):
    """
    if use_tpu = True: num_passes > 1 
    
    Since TPU training requires entire [bsz x tgt_len] chunks, it can discard
    as many as `bsz * tgt_len` tokens in training. When `bsz` and `tgt_len` are 
    both large, as in the case of TPU training for Transformer-XL, the problem
    may lead to detectable performance drop. 

    Here, we use multiple randomly shifted copies to deal with this problem.
    """

    num_step = len(data) // batch_size
    data = data[:batch_size * num_step]
    data = data.reshape(batch_size, num_step)

    return data


def create_ordered_tfrecords(save_dir, basename, data, batch_size, tgt_len):

    file_name = "{}.bsz-{}.tlen-{}.tfrecords".format(basename, batch_size, tgt_len)

    save_path = os.path.join(save_dir, file_name)
    record_writer = tf.io.TFRecordWriter(save_path)

    batched_data = batchify(data, batch_size)

    num_batch = 0
    # for t in range(0, batched_data.shape[1] - tgt_len - 1, tgt_len):
    for t in range(0, batched_data.shape[1] - 1, tgt_len):
        cur_tgt_len = min(batched_data.shape[1] - 1 - t, tgt_len)
        # drop the remainder if use tpu
        if num_batch % 500 == 0:
            print("  processing batch {}".format(num_batch))
        for idx in range(batch_size):
            inputs = batched_data[idx, t:t + cur_tgt_len]
            labels = batched_data[idx, t + 1:t + cur_tgt_len + 1]

            # features dict
            feature = {
                "inputs": _int64_feature(inputs),
                "labels": _int64_feature(labels),
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            record_writer.write(example.SerializeToString())

        num_batch += 1

    record_writer.close()
    print("Done writing {}. batches: {}".format(file_name, num_batch))

    return file_name, num_batch


def get_lm_corpus(data_dir, dataset):
    fn = os.path.join(data_dir, "cache.pkl")

    if exists(fn):
        print("Loading cached dataset...")
        with open(fn, "rb") as fp:
            corpus = pickle.load(fp)
    else:
        print("Producing dataset...")
        kwargs = {}
        kwargs["special"] = ["<eos>"]
        kwargs["lower_case"] = False

        corpus = Corpus(data_dir, dataset, **kwargs)

        print("Saving dataset...")
        with open(fn, "wb") as fp:
            pickle.dump(corpus, fp, protocol=2)

        corpus_info = {
            "vocab_size": len(corpus.vocab),
            "cutoffs": corpus.cutoffs,
            "dataset": corpus.dataset
        }
        with open(os.path.join(data_dir, "corpus-info.json"), "w") as fp:
            json.dump(corpus_info, fp)

    return corpus


def main(unused_argv):
    del unused_argv  # Unused

    corpus = get_lm_corpus(FLAGS.data_dir, FLAGS.dataset)

    save_dir = os.path.join(FLAGS.data_dir, "tfrecords")
    if not exists(save_dir):
        makedirs(save_dir)

    # save vocabulary for printing
    vocab_path = os.path.join(save_dir, "cache_vocab.pkl")
    with open(vocab_path, "wb") as fp:
        pickle.dump(corpus.vocab, fp, protocol=2)

    # test mode
    if FLAGS.per_host_test_bsz > 0:
        corpus.convert_to_tfrecords("test", save_dir, FLAGS.per_host_test_bsz, FLAGS.tgt_len, FLAGS.num_core_per_host, FLAGS=FLAGS)
        return

    for split, batch_size in zip(["train", "valid"], [FLAGS.per_host_train_bsz, FLAGS.per_host_valid_bsz]):
        if batch_size <= 0:
            continue
        print("Converting {} set...".format(split))
        corpus.convert_to_tfrecords(split, save_dir, batch_size, FLAGS.tgt_len, FLAGS.num_core_per_host, FLAGS=FLAGS)


def load_record_info(record_info_dir, split, per_host_bsz, tgt_len):
    record_name = "record_info-{}.bsz-{}.tlen-{}.json".format(split, per_host_bsz, tgt_len)

    record_info_path = os.path.join(record_info_dir, record_name)
    with open(record_info_path, "r") as fp:
        record_info = json.load(fp)

    return record_info


def get_input_fn(record_info_dir, split, per_host_bsz, tgt_len, num_core_per_host, num_hosts=1):
    """Creates input function."""
    record_info = load_record_info(record_info_dir, split, per_host_bsz, tgt_len)

    file_names = record_info["filenames"]
    num_batch = record_info["num_batch"]

    tf.compat.v1.logging.info("[{}] File names {}".format(split, file_names))

    def input_fn(params):
        # per-core batch size
        per_core_bsz = params["batch_size"]

        # data_dir could be a remote path, e.g., a google storage url
        data_dir = params["data_dir"]

        def parser(record):
            # whether allow the last batch with a potentially shorter length

            record_spec = {
                "inputs": tf.VarLenFeature(tf.int64),
                "labels": tf.VarLenFeature(tf.int64),
            }

            # retrieve serialized example
            example = tf.parse_single_example(
                serialized=record,
                features=record_spec)

            # cast int64 into int32
            # cast sparse to dense
            for key in list(example.keys()):
                val = example[key]
                if tf.keras.backend.is_sparse(val):
                    val = tf.sparse.to_dense(val)
                if val.dtype == tf.int64:
                    # val = tf.to_int32(val)
                    val = tf.cast(val, tf.int32)
                example[key] = val

            return example["inputs"], example["labels"]

        file_paths = []
        for file_name in file_names:
            file_path = os.path.join(data_dir, file_name)
            file_paths.append(file_path)

        if split == "train":
            dataset = tf.data.Dataset.from_tensor_slices(file_paths)
            if len(file_paths) > 1:
                dataset = dataset.shuffle(len(file_paths)).repeat()
                dataset = tf.data.TFRecordDataset(dataset)
            elif num_hosts > 1:
                host_id = params["context"].current_host
                # drop the remaining batches
                num_batch_per_host = num_batch // num_hosts

                my_start_sample_id = (host_id * num_batch_per_host * num_core_per_host * per_core_bsz)
                my_sample_num = num_batch_per_host * num_core_per_host * per_core_bsz
                dataset = tf.data.TFRecordDataset(dataset).skip(my_start_sample_id).take(my_sample_num)
            else:
                dataset = tf.data.TFRecordDataset(dataset)

            dataset = dataset.map(parser).cache().repeat()
            dataset = dataset.batch(per_core_bsz, drop_remainder=True)
            dataset = dataset.prefetch(num_core_per_host * per_core_bsz)
        else:
            # do not shuffle, repeat or cache in evaluation
            dataset = tf.data.Dataset.from_tensor_slices(file_paths)
            dataset = tf.data.TFRecordDataset(dataset)
            dataset = dataset.map(parser)
            dataset = dataset.batch(per_core_bsz, drop_remainder=True)

        return dataset

    if split == "train" and num_hosts > 1:
        record_info["num_batch"] = num_batch // num_hosts

    return input_fn, record_info


def get_corpus_info(corpus_info_path):
    with open(corpus_info_path, "r") as fp:
        corpus_info = json.load(fp)
    return corpus_info


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_string("data_dir", None, help="Location of the data corpus")
    flags.DEFINE_enum("dataset", "cdata", ["ptb", "wt2", "wt103", "lm1b", "enwik8", "text8", "cdata"], help="Dataset name.")
    flags.DEFINE_integer("per_host_train_bsz", 60, help="train batch size each host")
    flags.DEFINE_integer("per_host_valid_bsz", 60, help="valid batch size each host")
    flags.DEFINE_integer("per_host_test_bsz", 0, help="If > 0, enter test mode and process test set only. Otherwise, process train and dev sets only.")
    flags.DEFINE_integer("tgt_len", 70, help="number of tokens to predict")
    flags.DEFINE_integer("max_batch", -1, help="run in debug mode")
    flags.DEFINE_integer("num_core_per_host", 1, help="8 for TPU v2.")
    flags.DEFINE_bool("debug", default=False, help="Process only the first batch without shuffle for lm1b.")
    flags.DEFINE_integer("num_procs", 1, help="number of processes")
    flags.DEFINE_integer("num_shuffle", 4, help="number of shuffles for lm1b")

    tf.compat.v1.app.run(main)
