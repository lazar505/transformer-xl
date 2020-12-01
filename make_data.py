from absl import flags
import io
import numpy as np
import os
import spacy
import sys
from zipfile import ZipFile


FLAGS = flags.FLAGS
flags.DEFINE_string("raw_data_zip", default="cases.zip",
                    help="Path to input data directory.")
flags.DEFINE_string("data_dir", default="",
                    help="Path to output data directory.")
FLAGS(sys.argv)

if not os.path.exists(FLAGS.data_dir):
    print("Creating data directory: " + FLAGS.data_dir)
    os.makedirs(FLAGS.data_dir)

zfile = ZipFile(FLAGS.raw_data_zip)

members = zfile.namelist()
NUM_CASES = len(members)
print(NUM_CASES)
NUM_CASES = 50000
print(NUM_CASES)
members = np.random.permutation(members)[:NUM_CASES]
members_train = members[:int(NUM_CASES * 0.98)]
members_valid = members[int(NUM_CASES * 0.98):int(NUM_CASES * 0.99)]
members_test = members[int(NUM_CASES * 0.99):]

ITER_PRINT = NUM_CASES / 100
ITER_PRINT_TEST = NUM_CASES / 1000

print("Initializing spaCy")
nlp = spacy.load('en_core_web_lg')

print("========== TRAIN ==========")
file_path = os.path.join(FLAGS.data_dir, "train.txt")
file_o = io.open(file_path, "w+", encoding='utf8')
iteration = 1
for case in members_train:
    with zfile.open(case) as f:
        raw_text = f.read().decode('utf8')
    doc = nlp(raw_text)
    text_out = []
    for sent in doc.sents:
        for token in sent:
            text_out.append(token.text)
    text_out = u" " + ' '.join(text_out)
    while text_out.count(u"  ") > 0:
        text_out = text_out.replace(u"  ", u" ")
    file_o.write(text_out)
    if iteration % ITER_PRINT == 0:
        print("Iteration: " + str(iteration) + " File: " + str(case))
    iteration = iteration + 1
file_o.close()

print("========== VALIDATION ==========")
file_path = os.path.join(FLAGS.data_dir, "valid.txt")
file_o = io.open(file_path, "w+", encoding='utf8')
iteration = 1
for case in members_valid:
    with zfile.open(case) as f:
        raw_text = f.read().decode('utf8')
    doc = nlp(raw_text)
    text_out = []
    for sent in doc.sents:
        for token in sent:
            text_out.append(token.text)
    text_out = u" " + ' '.join(text_out)
    while text_out.count(u"  ") > 0:
        text_out = text_out.replace(u"  ", u" ")
    file_o.write(text_out)
    if iteration % ITER_PRINT_TEST == 0:
        print("Iteration: " + str(iteration) + " File: " + str(case))
    iteration = iteration + 1
file_o.close()

print("========== TEST ==========")
file_path = os.path.join(FLAGS.data_dir, "test.txt")
file_o = io.open(file_path, "w+", encoding='utf8')
iteration = 1
for case in members_test:
    with zfile.open(case) as f:
        raw_text = f.read().decode('utf8')
    doc = nlp(raw_text)
    text_out = []
    for sent in doc.sents:
        for token in sent:
            text_out.append(token.text)
    text_out = u" " + ' '.join(text_out)
    while text_out.count(u"  ") > 0:
        text_out = text_out.replace(u"  ", u" ")
    file_o.write(text_out)
    if iteration % ITER_PRINT_TEST == 0:
        print("Iteration: " + str(iteration) + " File: " + str(case))
    iteration = iteration + 1
file_o.close()
