import json
import os
import tensorflow as tf
from official.nlp import bert

import official.nlp.bert.tokenization


class Process(object):
    def __init__(self):
        self._tokenizer = bert.tokenization.FullTokenizer(
            vocab_file=os.path.join('vocab.txt'),
            do_lower_case=True)

    def _tokenizer_f(self, s):
        tokens = list(self._tokenizer.tokenize(s))
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def _get_input(self, texts):
        sentence = tf.ragged.constant([self._tokenizer_f(s)
                                       for s in texts])
        cls = [self._tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence.shape[0]
        input_word_ids = tf.concat([cls, sentence], axis=-1)

        input_mask = tf.ones_like(input_word_ids)

        type_cls = tf.zeros_like(cls)
        type_s = tf.zeros_like(sentence)

        input_type_ids = tf.concat(
            [type_cls, type_s], axis=-1)
        return input_word_ids, input_mask, input_type_ids

    def get_input_tensor(self, texts):
        """
        获得训练时需要的数据
        :param texts:
        :return:
        """
        input_word_ids, input_mask, input_type_ids = self._get_input(texts)
        inputs = {
            'input_word_ids': input_word_ids.to_tensor(),
            'input_mask': input_mask.to_tensor(),
            'input_type_ids': input_type_ids.to_tensor()}
        return inputs

    def request_json(self, texts, signature_name="serving_default"):
        """
        获得REST请求时需要的JSON
        :param texts:
        :param signature_name:
        :return:
        """
        print(texts)
        input_word_ids, input_mask, input_type_ids = self._get_input(texts)
        data = [{'input_word_ids': items[0],
                 'input_mask': items[1],
                 'input_type_ids': items[2]}
                for items in zip(input_word_ids.numpy().tolist(),
                                 input_mask.numpy().tolist(),
                                 input_type_ids.numpy().tolist())]
        print(data)
        return json.dumps({"signature_name": signature_name, "instances": data})
