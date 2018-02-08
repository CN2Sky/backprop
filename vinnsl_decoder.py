import json

def parse_vinnsl(vinnsl):

    nn_structure = {}

    parsed_json = json.loads(vinnsl)
    parameters = parsed_json['parameters']['input']
    structure = parsed_json['structure']

    learning_rate = parameters[0]['defaultValue']
    biasInput = parameters[1]['defaultValue']
    biasHidden = parameters[2]['defaultValue']
    momentum = parameters[3]['defaultValue']
    activationFunctionOutput = parameters[4]['defaultValue']
    activationFunctionHidden = parameters[5]['defaultValue']
    threshold = parameters[6]['defaultValue']
    target_data = parameters[7]['defaultValue']
    number_epochs = parameters[8]['defaultValue']

    connections = parsed_json['connections']

    fully_connected = connections['fullyConnected']['isConnected']
    shortcuts = connections['shortcuts']
    shortcuts_connections = shortcuts['connections']

    print(fully_connected)

    input_layer = structure['inputLayer']
    input_neurons = input_layer['amount']

    outputLayer = structure['outputLayer']
    output_neurons = outputLayer['amount']

    hidden_layers = structure['hiddenLayer']
    hidden_layers_neurons = []

    for layer in hidden_layers:
        hidden_layers_neurons.append(layer['amount'])

    nn_structure['input_neurons'] = input_neurons
    nn_structure['output_neurons'] = output_neurons
    nn_structure['hidden_layers'] = hidden_layers_neurons

    nn_structure['learning_rate'] = learning_rate
    nn_structure['biasInput'] = biasInput
    nn_structure['biasHidden'] = biasHidden
    nn_structure['momentum'] = momentum
    nn_structure['activationFunctionOutput'] = activationFunctionOutput
    nn_structure['activationFunctionHidden'] = activationFunctionHidden
    nn_structure['threshold'] = threshold
    nn_structure['target_data'] = target_data
    nn_structure['number_epochs'] = number_epochs

    print(nn_structure)

    return nn_structure