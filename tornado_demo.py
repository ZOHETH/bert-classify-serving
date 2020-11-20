from abc import ABC

import tornado.ioloop
import tornado.web
from tornado.httpserver import HTTPServer
import json
import base64
from datetime import datetime
import argparse
import requests
from bert_process import Process
import numpy as np

bert_pro = Process()
MAX_PROCESS = 1


def predict(texts):
    data = bert_pro.request_json(texts)
    json_response = requests.post('http://123.125.8.43:8501/v1/models/bert_model:predict', data=data)

    predictions = json.loads(json_response.text)['predictions']  # list
    predictions = np.argmax(predictions, axis=-1)
    return predictions


class TextHandler(tornado.web.RequestHandler, ABC):
    def post(self, **kwargs):
        the_time = self.request.headers['time']
        texts = self.request.arguments['texts'][0].decode()
        texts = texts.split('-')
        print("time:{}, texts:{}".format(the_time, texts))
        res = predict(texts)
        return self.write(bytes(json.dumps({"type": "TextHandler", "result": "ok", "data": res.tolist()}), 'UTF-8'))


def configure_app():
    return tornado.web.Application([
        ("/TextHandler", TextHandler),
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', help='Port Number', default=8888)
    args = parser.parse_args()
    print("Configure: {}".format(args))
    print("Server Start!!")

    app = configure_app()
    server = HTTPServer(app)
    server.bind(args.port)
    server.start(MAX_PROCESS)  # forks one process per cpu 0
    tornado.ioloop.IOLoop.current().start()
