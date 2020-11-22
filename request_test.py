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
import redis
from threading import Timer

from bert_process import Process


def demo(url):
    url = '{}/{}'.format(url, 'TextHandler')
    date = str(datetime.datetime.now())
    start = time.time()
    r = requests.post(url, headers={'time': date}, data={'texts': '我杀父查岗苏晨'}, timeout=5)
    end=time.time()
    print(r.text)
    end2 = time.time()
    print('time cost', end - start, 's')
    print('time cost', end2 - start, 's')


def test1():
    pro = Process()
    time_start = time.time()

    data = pro.request_json(['好贵啊', '我的'])

    headers = {"content-type": "application/json"}

    json_response = requests.post('http://123.125.8.43:8501/v1/models/bert_model:predict', data=data)
    time_end = time.time()

    predictions = json.loads(json_response.text)['predictions']  # list
    predictions = np.argmax(predictions, axis=-1)
    print(predictions)
    print('time cost', time_end - time_start, 's')


bert_pro = Process()
MAX_PROCESS = 1

if __name__ == "__main__":
    demo("http://localhost:8888")
    # test1()
