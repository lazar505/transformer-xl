from absl import flags
import io
import os
import pickle
import sys


FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default="",
                    help="Path to data directory.")
FLAGS(sys.argv)

vocab_path = os.path.join(FLAGS.data_dir, "cache_vocab.pkl")
print("Loading cached vocab...")
with open(vocab_path, "rb") as fp:
    vocab = pickle.load(fp)

print("Writing vocab tokens...")
vocab_tokens_path = os.path.join(FLAGS.data_dir, "vocab_tokens.txt")
file_o = io.open(vocab_tokens_path, "w", encoding='utf8')
file_o.write("Number of tokens " + str(len(vocab)) + "\n" + "\n")
for i in range(len(vocab)):
    file_o.write(str(i) + " " + vocab.get_sym(i) + "\n")
file_o.close()
