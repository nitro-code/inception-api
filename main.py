#!/usr/bin/env python
from flask import Flask, request
import urllib2 as urllib
import json

from PIL import Image
from cStringIO import StringIO

import numpy as np
import model


app = Flask(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

@app.route('/api/v1/predict', methods=['GET'])
def inference():
    url = request.args.get('url', '')
    url_decoded = urllib.unquote(url).decode('utf8')

    img_handle = urllib.urlopen(url_decoded)
    img_data = StringIO(img_handle.read())

    pil_img = Image.open(img_data)
    pil_img = pil_img.resize((299, 299), Image.ANTIALIAS)

    predictions = model.predict_pil(pil_img)
    return json.dumps(predictions, cls=NumpyEncoder)
