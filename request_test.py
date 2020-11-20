import requests
import json
import numpy as np
import time

from tornado.ioloop import IOLoop, PeriodicCallback
from tornado import gen
from tornado.websocket import websocket_connect
import sys
import requests
import datetime
import json
import base64
import argparse

from bert_process import Process


def demo(url):
    url = '{}/{}'.format(url, 'TextHandler')
    date = str(datetime.datetime.now())
    start = time.time()
    texts = ['好贵啊', '我的头']
    data = '-'.join(texts)
    r = requests.post(url, headers={'time': date}, data={'texts': data}, timeout=5)
    end = time.time()
    j = json.loads(r.text)
    print('Recv: {}:{}'.format(j['type'], j['result']))
    print('Echo data: {}'.format(j['data']))
    end2 = time.time()
    print('time cost', end - start, 's')
    print('time cost', end2 - start, 's')


def test1():
    pro = Process()
    time_start = time.time()

    data = pro.request_json(['好贵啊', '我的头'])

    headers = {"content-type": "application/json"}

    json_response = requests.post('http://123.125.8.43:8501/v1/models/bert_model:predict', data=data)
    time_end = time.time()

    predictions = json.loads(json_response.text)['predictions']  # list
    predictions = np.argmax(predictions, axis=-1)
    print(predictions)
    print('time cost', time_end - time_start, 's')


# data = json.dumps(['好贵啊', '我的头'])
# json_response = requests.post('http://hostlocal:8889/chatsocket', data=data)
# print(json_response.text)

if __name__ == "__main__":
    demo("http://localhost:8888")
    # test1()
