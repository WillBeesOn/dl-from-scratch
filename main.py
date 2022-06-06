import argparse
from layers import InputLayer, FullyConnected, SigmoidLayer, ReLuLayer, TanhLayer, SoftmaxLayer, LinearLayer
from objectives import LogLoss, CrossEntropy, LeastSquares
from utils import transpose, save_csv, load_csv

parser = argparse.ArgumentParser(description='Trains on minst data with various architectures.')
parser.add_argument('-t', '--train', type=str, metavar='path', default='mnist_train_100.csv',
                    help='Path to training data. (default: mnist_train_100)')
parser.add_argument('-v', '--validate', type=str, metavar='path', default='mnist_valid_10.csv',
                    help='Path to validation data. (default: mnist_valid_10.csv)')
parser.add_argument('-e', '--epochs', type=int, metavar='num', default=5,
                    help='Number of epochs to run for part 2 and 3. (default: 5)')
parser.add_argument('-l', '--l_rate', type=float, metavar='num', default=0.1,
                    help='Learning rate to use. (default: 0.1)')
args = parser.parse_args()


# Do forward prop to determine accuracy and also classify.
def forward(layers, data, y, y_hot):
    f_data = data
    num_layers = len(layers)
    objective = 0
    accuracy = 0
    for i, layer in enumerate(layers):
        # If this is the last layer, it's the objective function. Test classifying.
        if i == num_layers - 1:
            correct = 0
            if isinstance(layer, LeastSquares):
                for (expected, calculated) in zip(y, transpose(f_data)[0]):
                    r = round(calculated)
                    if r == expected:
                        correct += 1
                accuracy = correct * 100 / len(y)
                objective = layer.eval(transpose(y), f_data)
            else:
                for (expected, calculated) in zip(y, [row.index(max(row)) for row in f_data]):
                    if calculated == expected:
                        correct += 1
                accuracy = correct * 100 / len(y_hot)
                objective = layer.eval(y_hot, f_data)
        else:
            f_data = layer.forward(f_data)
    return f_data, objective, accuracy


# Do the training, taking the data and doing forward prop then back prop to learn weights.
def train_model(layers, train, train_y, train_yhot, test, test_y, test_yhot, epochs=args.epochs):
    outputs_t = []
    outputs_v = []
    accuracies_t = []
    accuracies_v = []

    for e in range(1, epochs + 1):
        print(f'Running epoch {e}...')
        # Test classification on validation data first so it doesn't disrupt learning on training data
        f_test, out, acc = forward(layers, test, test_y, test_yhot)
        outputs_v.append(out)
        accuracies_v.append(acc)

        # Forward prop the training data. Also get cross entropy output and classification accuracy for this epoch
        f, out, acc = forward(layers, train, train_y, train_yhot)
        outputs_t.append(out)
        accuracies_t.append(acc)

        # Backward prop
        b_data = f
        for i, layer in enumerate(reversed(layers)):
            # If this is the first layer, then this is our output layer. Need to pass expected value to it too.
            if i == 0:
                b_data = layer.backward(transpose(train_y) if isinstance(layer, LeastSquares) else train_yhot, b_data)
            elif isinstance(layer, FullyConnected):
                # If we are at a fully connected layer, don't backprop, just update weights and break
                layer.updateWeights(b_data, epoch=e, l_rate=args.l_rate, adam=True)

                if i == len(layers) - 2:
                    break
            else:
                b_data = layer.backward(b_data)

    # Accuracies are calculated before backprop, so we need to predict 1 more time to account for final backprop
    f_test, out, acc = forward(layers, test, test_y, test_yhot)
    outputs_v.append(out)
    accuracies_v.append(acc)

    f, out, acc = forward(layers, train, train_y, train_yhot)
    outputs_t.append(out)
    accuracies_t.append(acc)

    return outputs_t, accuracies_t, outputs_v, accuracies_v


# Prep train data. Remove first column which is the expected classification.
train = load_csv(args.train)
train_t = transpose(train)

# One hot encoding for y values
train_y = train_t.pop(0)
train_yhot = [[1 if i == y_class else 0 for i in range(0, 10)] for y_class in train_y]
train = transpose(train_t)

# Prep validation data. Remove first column which is the expected classification.
validate = load_csv(args.validate)
validate_t = transpose(validate)

# One hot encoding for y values
validate_y = validate_t.pop(0)
validate_yhot = [[1 if i == y_class else 0 for i in range(0, 10)] for y_class in validate_y]
validate = transpose(validate_t)

# First hidden layers (fully connected --> other activation) are different per architecture.
feat_num = len(train[0])
architectures = [
    [
        InputLayer(train),
        FullyConnected(len(train[0]), 10),
        SigmoidLayer(),
        LogLoss()
    ],
    [
        InputLayer(train),
        FullyConnected(feat_num, feat_num),
        TanhLayer(),
        FullyConnected(feat_num, 10),
        SigmoidLayer(),
        LogLoss()
    ],
    [
        InputLayer(train),
        FullyConnected(feat_num, feat_num),
        TanhLayer(),
        FullyConnected(feat_num, 10),
        SoftmaxLayer(),
        CrossEntropy()
    ]
]

for i, arch in enumerate(architectures):
    print(f'Learning on mnist data with architecture {i + 1}...')
    ot, at, ov, av = train_model(arch, train, train_y, train_yhot, validate, validate_y, validate_yhot)
    spaces = ['' for i in range(len(ot))]

    save_csv(f'output/arch{i + 1}_train_and_validate.csv', transpose([ot, at, spaces, ov, av]))
    print('Saved training and validate output data.')