import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import io

from keras.models import load_model

from nn import nn
import vinnsl_decoder
import numpy as np
import subprocess
from flask import Flask
from flask import request
import csv
import json


app = Flask(__name__)


@app.route('/train', methods=['GET', 'POST'])
def train():
    vinnsl_description = request.form['vinnsl']
    training_data = request.form['training_data']
    model_id = request.form['model_id']

    description = vinnsl_decoder.parse_vinnsl(vinnsl_description)

    training_data = np.array(eval(training_data), "float32")
    # target_data = np.array(eval(target_data), "float32")
    model = nn.train_model(training_data, description, model_id)
    model.save('models/my_model.h5')
    proc = subprocess.Popen(['python', 'serialization/encoder.py', 'models/my_model.h5'], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return proc.communicate()[0]


@app.route('/logs/<model_id>', methods=['GET'])
def logs(model_id):

    csvfile = open('logs/' + model_id, 'r')
    reader = csv.DictReader(csvfile)

    out = json.dumps([row for row in reader])

    print(out)

    return out, 200, {'Content-Type': 'text/css; charset=utf-8'}


@app.route('/test', methods=['POST'])
def test():
    model = request.form['model']
    testing_data = request.form['testing_data']
    x = eval(testing_data)
    testing_data = np.array(x, "float32")

    with io.open('models/model.json', 'w', encoding='utf-8') as f:
        f.write(model)

    if os.path.exists('models/model.h5'):
        os.remove('models/model.h5')

    p = subprocess.Popen(['python', 'serialization/decoder.py', 'models/model.json', 'models/model.h5'],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    p_status = p.wait()

    model = load_model('models/model.h5')
    predictions = model.predict(testing_data).round()

    return str(predictions)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
