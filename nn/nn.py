import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras import optimizers
from keras.callbacks import RemoteMonitor
from keras.callbacks import CSVLogger


def train_model(training_data, testing_data, epoche, description, parameters):
    csv_logger = CSVLogger('data/training.log')

    #Sending realtime training data to visualization
    # remote = RemoteMonitor(root='http://localhost:9000')

    model = Sequential()

    dimensions = training_data[0].size

    input_layer = description['input_neurons']
    output_layer = description['output_neurons']
    hidden_layers = description['hidden_layers']

    learning_rate = float(description['learning_rate'])
    momentum = float(description['momentum'])
    activationFunctionOutput = description['activationFunctionOutput']
    activationFunctionHidden = description['activationFunctionHidden']

    model.add(Dense(input_layer, input_dim=dimensions, activation=activationFunctionHidden))
    for layer in hidden_layers:
        model.add(Dense(layer, input_dim=dimensions, activation=activationFunctionHidden))
    model.add(Dense(output_layer, activation=activationFunctionOutput))

    sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=momentum, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['binary_accuracy'])

    model.fit(training_data, testing_data, epochs=epoche, verbose=2, callbacks=[csv_logger])
    # model.fit(training_data, testing_data, epochs=epoche, verbose=2, callbacks=[remote])

    model.save('models/my_model.h5')

    return model