import tensorflow as tf
import tensorflow_datasets as tfds

from sklearn.metrics import accuracy_score

from preprocess import load_data
from bert_process import Process

tfds.disable_progress_bar()

X_train, X_val, X_test, y_train, y_val, y_test = load_data()

pro = Process()

export_dir = './saved_model_test/1'
reloaded = tf.saved_model.load(export_dir)

for i in range(0, len(y_test), 1000):
    text = pro.get_input_tensor(X_test[i:i + 1000])
    reloaded_result = reloaded([text['input_word_ids'],
                                text['input_mask'],
                                text['input_type_ids']], training=False)

    # The results are (nearly) identical:
    reloaded_result = tf.argmax(reloaded_result, axis=-1).numpy()
    print('acc:', accuracy_score(y_test[i:i + 1000], reloaded_result))
