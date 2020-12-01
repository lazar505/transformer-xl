from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import bottle
import io
import math
import numpy as np
import os
import pickle
import random
from scipy.special import softmax
import spacy
import sys
import tensorflow as tf

import data_utils
import model


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# GPU config
flags.DEFINE_integer("num_core_per_host", default=8,
                     help="Number of cores per host")

# Experiment (data/checkpoint/directory) config
flags.DEFINE_string("data_dir", default="",
                    help="Path to tf-records directory.")
flags.DEFINE_string("record_info_dir", default="",
                    help="Path to local directory containing filenames.txt.")
flags.DEFINE_string("corpus_info_path", default="",
                    help="Path to corpus-info.json file.")
flags.DEFINE_string("model_dir", default=None,
                    help="Estimator model_dir.")
flags.DEFINE_string("eval_ckpt_path", None,
                    help="Checkpoint path for do_test evaluation."
                         "If set, model_dir will be ignored."
                         "If unset, will use the latest ckpt in model_dir.")

# Training config
flags.DEFINE_integer("eval_batch_size", default=60,
                     help="Size of valid batch.")

# Model config
flags.DEFINE_integer("tgt_len", default=70,
                     help="Number of steps to predict")
flags.DEFINE_integer("mem_len", default=70,
                     help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False,
                  help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1,
                     help="Clamp length")

flags.DEFINE_integer("n_layer", default=6,
                     help="Number of layers.")
flags.DEFINE_integer("d_model", default=500,
                     help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=500,
                     help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=10,
                     help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=50,
                     help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=1000,
                     help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.1,
                   help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
                   help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False,
                  help="untie r_w_bias and r_r_bias")

# Adaptive Softmax / Embedding
flags.DEFINE_integer("div_val", default=1,
                     help="Divide the embedding size by this val for each bin")
flags.DEFINE_bool("proj_share_all_but_first", default=False,
                  help="True to share all but first projs, False not to share.")
flags.DEFINE_bool("proj_same_dim", default=True,
                  help="Project the bin with the same dimension.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
                  enum_values=["normal", "uniform"],
                  help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
                   help="Initialization std when init is normal.")
flags.DEFINE_float("proj_init_std", default=0.01,
                   help="Initialization std for embedding projection.")
flags.DEFINE_float("init_range", default=0.1,
                   help="Initialization std when init is uniform.")


FLAGS = flags.FLAGS
FLAGS(sys.argv)


def get_model_fn(n_token, cutoffs):
    def model_fn(inp, tgt, mems, is_training, generate_text, prev_text):
        if generate_text == False:
            inp = tf.transpose(inp, [1, 0])
            tgt = tf.transpose(tgt, [1, 0])

        if FLAGS.init == "uniform":
            initializer = tf.initializers.random_uniform(
                minval=-FLAGS.init_range,
                maxval=FLAGS.init_range,
                seed=None)
        elif FLAGS.init == "normal":
            initializer = tf.initializers.random_normal(
                stddev=FLAGS.init_std,
                seed=None)
            proj_initializer = tf.initializers.random_normal(
                stddev=FLAGS.proj_init_std,
                seed=None)

        tie_projs = [False for _ in range(len(cutoffs) + 1)]
        if FLAGS.proj_share_all_but_first:
            for i in range(1, len(tie_projs)):
                tie_projs[i] = True

        loss, new_mems, t_output = model.transformer(
            dec_inp=inp,
            target=tgt,
            mems=mems,
            n_token=n_token,
            n_layer=FLAGS.n_layer,
            d_model=FLAGS.d_model,
            d_embed=FLAGS.d_embed,
            n_head=FLAGS.n_head,
            d_head=FLAGS.d_head,
            d_inner=FLAGS.d_inner,
            dropout=FLAGS.dropout,
            dropatt=FLAGS.dropatt,
            initializer=initializer,
            proj_initializer=proj_initializer,
            is_training=is_training,
            generate_text=generate_text,
            prev_text=prev_text,
            mem_len=FLAGS.mem_len,
            cutoffs=cutoffs,
            div_val=FLAGS.div_val,
            tie_projs=tie_projs,
            input_perms=None,
            target_perms=None,
            head_target=None,
            same_length=FLAGS.same_length,
            clamp_len=FLAGS.clamp_len,
            untie_r=FLAGS.untie_r,
            proj_same_dim=FLAGS.proj_same_dim)

        # number of parameters
        num_params = sum([np.prod(v.shape) for v in tf.compat.v1.trainable_variables()])
        tf.compat.v1.logging.info('#params: {}'.format(num_params))

        # format_str = '{{:<{0}s}}\t{{}}'.format(
        #     max([len(v.name) for v in tf.compat.v1.trainable_variables()]))
        # for v in tf.compat.v1.trainable_variables():
        #   tf.compat.v1.logging.info(format_str.format(v.name, v.get_shape()))

        if is_training:
            all_vars = tf.compat.v1.trainable_variables()
            grads = tf.gradients(loss, all_vars)
            grads_and_vars = list(zip(grads, all_vars))

            return loss, new_mems, grads_and_vars
        elif generate_text:
            return loss, new_mems, t_output
        else:
            return loss, new_mems, inp

    return model_fn


def single_core_graph(n_token, cutoffs, is_training, inp, tgt, mems, generate_text=False, prev_text=None):
    model_fn = get_model_fn(
        n_token=n_token,
        cutoffs=cutoffs)

    model_ret = model_fn(
        inp=inp,
        tgt=tgt,
        mems=mems,
        is_training=is_training,
        generate_text=generate_text,
        prev_text=prev_text)

    return model_ret

def generate_text(n_token, cutoffs, ps_device, vocab, text_length=100, word_range=3):
    per_core_bsz = FLAGS.eval_batch_size // FLAGS.num_core_per_host
    tower_mems, tower_losses, tower_new_mems = [], [], []

    for i in range(FLAGS.num_core_per_host):
        with tf.device("/device:CPU:0"), \
             tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=tf.AUTO_REUSE):
            mems_i = [tf.compat.v1.placeholder(tf.float32,
                                               [FLAGS.mem_len, per_core_bsz, FLAGS.d_model])
                      for _ in range(FLAGS.n_layer)]

            prev_text_tf = tf.compat.v1.placeholder(tf.int64,
                                                    [FLAGS.tgt_len, 1])

            loss_i, new_mems_i, t_output = single_core_graph(
                n_token=n_token,
                cutoffs=cutoffs,
                is_training=False,
                inp=None,
                tgt=None,
                mems=mems_i,
                generate_text=True,
                prev_text=prev_text_tf)

            tower_mems.append(mems_i)
            tower_losses.append(loss_i)
            tower_new_mems.append(new_mems_i)

    ## sum losses across towers
    if len(tower_losses) > 1:
        loss = tf.add_n(tower_losses) / len(tower_losses)
    else:
        loss = tower_losses[0]

    tower_mems_np = [
        [np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model], dtype=np.float32)
         for layer in range(FLAGS.n_layer)]
        for core in range(FLAGS.num_core_per_host)
    ]

    start_text = vocab.encode_file("temp_input_text.txt", ordered=True)

    prev_text_np = []
    start_en_dec = []

    for i in start_text[-FLAGS.tgt_len:]:
        prev_text_np.append([i])
        start_en_dec.append(vocab.get_sym(i))

    start_en_dec = ' '.join(start_en_dec)

    text = []

    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(allow_soft_placement=True, device_count={'GPU': 0})) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        if FLAGS.eval_ckpt_path is None:
            eval_ckpt_path = tf.compat.v1.train.latest_checkpoint(FLAGS.model_dir)
        else:
            eval_ckpt_path = FLAGS.eval_ckpt_path
        tf.compat.v1.logging.info("Model: {}".format(eval_ckpt_path))
        saver.restore(sess, eval_ckpt_path)

        fetches = [loss, tower_new_mems, t_output]

        print("Generating text...")

        step = 0
        while step < text_length:
            feed_dict = {}
            for i in range(FLAGS.num_core_per_host):
                for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
                    feed_dict[m] = m_np
                feed_dict[prev_text_tf] = prev_text_np

            fetched = sess.run(fetches, feed_dict=feed_dict)

            sm_prob = fetched[2]
            sm_prob = softmax(sm_prob[-1, 0, :])
            sorted_indices = np.argsort(sm_prob)

            # random word choice from the set of tokens with the highest probabilities
            # If we put the most probable word, repetition of the same phrases can occur
            # word_range = 3
            next_word_index = int(math.floor(random.uniform(1, word_range + 1)))
            next_word = sorted_indices[-next_word_index]
            text.append(vocab.get_sym(next_word))
            prev_text_np = np.append(prev_text_np[1:], [next_word]).reshape((FLAGS.tgt_len, 1))
            loss_np, tower_mems_np = fetched[:2]

            if vocab.get_sym(next_word) != "<eos>":
                step = step + 1

    stext = ' '.join(text)
    return start_en_dec, stext


input_page = """
<h3>
    <center>
    Legal Text Generation with Transformers
</h3>
<h4>
    <center>
    Lazar Peric, Stefan Mijic, Dominik Stammbach and Elliott Ash<br>
    ETH Zurich<br>
</h4>

<form action="/transformer" method="post">
    Text length: <input name="text_length" type="text" value="100" />
    Randomness: <input name="randomness" type="text" value="3" /><br><br>
    Input text:<br>
    <textarea rows="18" style="width:90%;" name="input_text">Law is a bottomless pit.</textarea><br>
    <input value="Generate" type="submit" />
</form>
"""

output_page = """
<h3>
    <center>
    Legal Text Generation with Transformers
</h3>
<h4>
    <center>
    Lazar Peric, Stefan Mijic, Dominik Stammbach and Elliott Ash<br>
    ETH Zurich<br>
</h4>
Input text:<br>
{{input_text}}<br><br>
Output text:<br>
{{output_text}}
"""


@bottle.get('/transformer')
def get_transformer_text():
    return input_page


@bottle.post('/transformer')
def generate_new_text():
    original_text = bottle.request.forms.get('input_text')

    def process_text(text, process=False):
        if process:
            text = text.replace(u" , ", u", ")
            text = text.replace(u" . ", u". ")
            text = text.replace(u" ; ", u"; ")
            text = text.replace(u" : ", u": ")
            text = text.replace(u" ( ", u" (")
            text = text.replace(u" ) ", u") ")
            text = text.replace(u"<eos>", u" ")
            text = text.replace(u" 's ", u"'s ")
            while text.count(u"  ") > 0:
                text = text.replace(u"  ", u" ")
        return text

    text_length = abs(int(bottle.request.forms.get("text_length")))
    word_range = abs(int(bottle.request.forms.get("randomness")))

    if word_range == 0:
        word_range = 1

    print("Processing text...")
    doc = nlp(original_text)
    text_out = []
    for sent in doc.sents:
        for token in sent:
            text_out.append(token.text)
    if len(text_out) < test_tgt_len:
        text_out = ["<unk>"] * (test_tgt_len - len(text_out)) + text_out
    text_out = u" " + ' '.join(text_out)
    while text_out.count(u"  ") > 0:
        text_out = text_out.replace(u"  ", u" ")

    file_o = io.open("temp_input_text.txt", "w", encoding='utf8')
    file_o.write(text_out)
    file_o.close()

    input_text, output_text = generate_text(n_token, cutoffs, "/gpu:0", vocab, text_length=text_length,
                                            word_range=word_range)

    os.remove("temp_input_text.txt")

    input_text = process_text(original_text, process=True)
    output_text =process_text(output_text, process=True)
    return bottle.template(output_page, input_text=original_text, output_text=output_text)

def main(unused_argv):
    del unused_argv  # Unused
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # chech: http://localhost:8787/transformer
    bottle.run(host='0.0.0.0', port=8787, reloader=True)


print("Loading Spacy...")
nlp = spacy.load('en_core_web_lg')
print("Loading corpus info...")
corpus_info = data_utils.get_corpus_info(FLAGS.corpus_info_path)
n_token = corpus_info["vocab_size"]
cutoffs = corpus_info["cutoffs"][1:-1]
test_tgt_len = FLAGS.tgt_len
vocab_path = os.path.join(FLAGS.data_dir, "cache_vocab.pkl")
print("Loading cached vocabulary...")
with open(vocab_path, "rb") as fp:
    vocab = pickle.load(fp)

tf.compat.v1.app.run()
