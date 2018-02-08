import io

from keras.models import load_model

from nn import nn
import vinnsl_decoder
import numpy as np
import os
import h5json
import subprocess
from flask import Flask
from flask import request

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

@app.route('/train', methods=['GET','POST'])
def train():

    vinnsl_description = request.form['vinnsl']
    training_data = request.form['training_data']
    epoche = int(request.form['epoche'])
    target_data = request.form['target_data']

    description = vinnsl_decoder.parse_vinnsl(vinnsl_description)
    parameters = []

    training_data = np.array(eval(training_data), "float32")
    target_data = np.array(eval(target_data), "float32")
    model = nn.train_model(training_data, target_data, epoche, description, parameters)
    model.save('models/my_model.h5')
    proc = subprocess.Popen(['python', 'serialization/encoder.py', 'models/my_model.h5'], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return proc.communicate()[0]

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

    p = subprocess.Popen(['python', 'serialization/decoder.py', 'models/model.json', 'models/model.h5'], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    p_status = p.wait()

    model = load_model('models/model.h5')
    predictions = model.predict(testing_data).round()

    return str(predictions)

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=5000)
