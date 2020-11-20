import os
import json
import numpy as np
import matplotlib.pyplot as plt
import collections

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
from preprocess import load_data
from tokenizer import FullTokenizer

from bert_process import Process

tfds.disable_progress_bar()

X_train, X_val, X_test, y_train, y_val, y_test = load_data()
# X_train, y_train = X_train[:100], y_train[:100]
# X_val, y_val = X_val[:20], y_val[:20]
gs_folder_bert = 'model1'
print(tf.io.gfile.listdir(gs_folder_bert))

bert_pro=Process()


text_train = bert_pro.get_input_tensor(X_train)

text_val = bert_pro.get_input_tensor(X_val)

text_test = bert_pro.get_input_tensor(X_test)

for key, value in text_train.items():
    print(f'{key:15s} shape: {value.shape}')

print(f'text_train_labels shape: {y_train.shape}')

bert_config_file = os.path.join(gs_folder_bert, "bert_config.json")
config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())

bert_config = bert.configs.BertConfig.from_dict(config_dict)

bert_classifier, bert_encoder = bert.bert_models.classifier_model(
    bert_config, num_labels=41)

tf.keras.utils.plot_model(bert_classifier, show_shapes=True, dpi=48)

glue_batch = {key: val[:10] for key, val in text_train.items()}

checkpoint = tf.train.Checkpoint(model=bert_encoder)
checkpoint.restore(
    os.path.join(gs_folder_bert, 'model1-1')).assert_consumed()
print(checkpoint)

print(bert_classifier(
    glue_batch, training=True
).numpy())

# Set up epochs and steps
epochs = 3
batch_size = 64
eval_batch_size = 64

train_data_size = len(y_train)
steps_per_epoch = int(train_data_size / batch_size)
num_train_steps = steps_per_epoch * epochs
warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

# creates an optimizer with learning rate schedule
optimizer = nlp.optimization.create_optimizer(
    2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

bert_classifier.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics)

bert_classifier.fit(
    text_train, y_train,
    validation_data=(text_val, y_val),
    batch_size=64,
    epochs=epochs)

# my_examples = bert_encode(['好贵啊'], tokenizer)
# result = bert_classifier(my_examples, training=False)
# result = tf.argmax(result,axis=-1).numpy()
# print(result)

call = tf.function(bert_classifier.call)
call = call.get_concrete_function(
    [tf.TensorSpec(shape=(None, None,), dtype=tf.int32, name='input_word_ids'),
     tf.TensorSpec(shape=(None, None,), dtype=tf.int32, name='input_mask'),
     tf.TensorSpec(shape=(None, None,), dtype=tf.int32, name='input_type_ids')
     ])

export_dir = './temp/1'
tf.saved_model.save(bert_classifier, export_dir=export_dir, signatures=call)

