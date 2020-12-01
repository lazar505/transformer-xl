# Text generation with Transformer-XL
Transformer-XL implementation in Python and TensorFlow from
https://github.com/kimiyoung/transformer-xl adapted for text generation.
The code allows training of the transformer on a custom dataset
and interactive text generation with the transformer using
command line interface or web browser.

## Setup

Run `bash install_venv.sh` to create a python virtual environment
and install required packages that are listed in `requirements.txt`.

Make following changes in `run_command.sh` file:
- Set `RAW_DATA` to a zip file path that contains the raw dataset.
- Set `DATA_ROOT` to a directory path where the dataset is going be stored.
- Set `MODEL_ROOT` to a directory path where the trained model is going be saved.
- Set the desired model parameters.

## Commands

Run `bash run_command.sh command_name` to execute a command with name `command_name`.
Possible options for `command_name` are:

- `make_data` - read the raw dataset and create `train.txt`, `valid.txt`, and `train.txt`
that contain the dataset.
- `train_data` or `test_data` - convert the dataset to tfrecords.
- `vocab` - print the vocabulary to a file.
- `train` - train the transformer.
- `eval` - evaluate the transformer.
- `gen` - start interactive text generation with the transformer.
- `web` - start a web server that provides interactive text generation.
- `make_experiment_data` - create a csv file with one column where each row
has text with specified number of sentences.
- `experiment` - generate continuations for the sentences in the csv file.
