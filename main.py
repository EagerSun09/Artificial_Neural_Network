# CS-131 Artificial Intelligence
# A6 - ANN
# Yige Sun

import math
import random
import copy

LEARNING_RATE = 0.1
BIAS = 1
VISIBLE = False
class Data:
    # data will be transformed in to [sepal_length, sepal_width, petal_length, petal width]
    # unit of length and width is cm
    def __init__(self, sepal_length, sepal_width, petal_length, petal_width, label):
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width

        self.setosa = 0
        self.versicolour = 0
        self.virginica = 0
        if label == "Iris-setosa":
            self.setosa = 1
        elif label == "Iris-versicolor":
            self.versicolour = 1
        elif label == "Iris-virginica":
            self.virginica = 1

        return

    def __str__(self):
        message = "["+str(self.sepal_length)+", "+str(self.sepal_width) + ", "+ str(self.petal_length) + ", "\
                  +str(self.petal_width)+", "+str(self.setosa)+", "+str(self.versicolour)+", "+str(self.virginica) + "]"
        return message

class ANN:
    def __init__(self):
        self.topology = [4, 8, 3]
        # size of input layer is 4
        # size of hidden layer is 5, I tried from 1 to 9, and 8 works best
        # size of output layer is 3
        self.INPUT_LAYER = 0
        self.OUTPUT_LAYER = 2
        self.SEPAL_LENGTH_INPUT = 0
        self.SEPAL_WIDTH_INPUT = 1
        self.PETAL_LENGTH_INPUT = 2
        self.PETAL_WIDTH_INPUT = 3
        self.SETOSA_OUTPUT = 0
        self.VERSICOLOUR_OUTPUT = 1
        self.VIRGINICA_OUTPUT = 2
        # parameters
        self.dev = 1
        self.learning_rate = LEARNING_RATE
        self.bias = BIAS
        # initiate bias weights in random fashion
        self.bias_weights = list()
        for layers in self.topology:
            layer_bias_weights = list()
            for neuron in range(0, layers):
                layer_bias_weights.append(random.uniform(-0.2, 0.2))
            self.bias_weights.append(layer_bias_weights)

        # potential of each layer is 0 now
        self.potentials = list()
        for layers in self.topology:
            layer_potentials = list()
            for neuron in range(0, layers):
                layer_potentials.append(0)
            self.potentials.append(layer_potentials)

        # initiate output and delta in same way as potentials
        self.outputs = copy.deepcopy(self.potentials)
        self.deltas = copy.deepcopy(self.outputs)

        # weights between neurons
        self.weights = list()
        for parent_layer in range(0, len(self.topology) - 1):
            layer_weights = []
            for parent_neuron in range(0, self.topology[parent_layer]):
                neuron_weights = []
                for child_neuron in range(0, self.topology[parent_layer + 1]):
                    neuron_weights.append(random.uniform(-0.2, 0.2))
                layer_weights.append(neuron_weights)
            self.weights.append(layer_weights)
        return

    def transformer(self, raw_date):
        data_file = open(raw_date)
        data = list()
        for line in data_file:
            raw = line.rstrip().split(",")
            if raw != ['']:
                sepal_length = float(raw[0])
                sepal_width = float(raw[1])
                petal_length = float(raw[2])
                petal_width = float(raw[3])
                label = raw[4]
                example = Data(sepal_length, sepal_width, petal_length, petal_width, label)
                data.append(example)
        return data

    def decorrelate(self, datas):
        total_num = len(datas)

        total_sepal_len = 0
        total_sepal_wid = 0
        total_petal_len = 0
        total_petal_wid = 0

        squared_sepal_len = 0
        squared_sepal_wid = 0
        squared_petal_len = 0
        squared_petal_wid = 0

        for data in datas:
            total_sepal_len += data.sepal_length
            total_sepal_wid += data.sepal_width
            total_petal_len += data.petal_length
            total_petal_wid += data.petal_width

            squared_sepal_len += data.sepal_length ** 2
            squared_sepal_wid += data.sepal_width ** 2
            squared_petal_len += data.petal_length ** 2
            squared_petal_wid += data.petal_width ** 2
        self.ave_sepal_len = total_sepal_len / total_num
        self.ave_sepal_wid = total_sepal_wid / total_num
        self.ave_petal_len = total_petal_len / total_num
        self.ave_petal_wid = total_petal_wid / total_num
        average_squared_sepal_length = squared_sepal_len / total_num
        average_squared_sepal_width = squared_sepal_wid / total_num
        average_squared_petal_length = squared_petal_len / total_num
        average_squared_petal_width = squared_petal_wid / total_num
        self.var_sepal_length = average_squared_sepal_length - self.ave_sepal_len * self.ave_sepal_len
        self.var_sepal_width = average_squared_sepal_width - self.ave_sepal_wid * self.ave_sepal_wid
        self.var_petal_length = average_squared_petal_length - self.ave_petal_len * self.ave_petal_len
        self.var_petal_width = average_squared_petal_width - self.ave_petal_wid * self.ave_petal_wid

        for data in datas:
            data.sepal_length = (data.sepal_length - self.ave_sepal_len) / self.var_sepal_length
            data.sepal_width = (data.sepal_width - self.ave_sepal_wid) / self.var_sepal_width
            data.petal_length = (data.petal_length - self.ave_petal_len) / self.var_petal_length
            data.petal_width = (data.petal_width - self.ave_petal_wid) / self.var_petal_width

        return datas

    def rescale(self, datas):
        # rescale data in min-max scaler
        max_sepal_len = datas[0].sepal_length
        min_sepal_len = datas[0].sepal_length

        max_sepal_wid = datas[0].sepal_width
        min_sepal_wid = datas[0].sepal_width

        max_petal_len = datas[0].petal_length
        min_petal_len = datas[0].petal_length

        max_petal_wid = datas[0].petal_width
        min_petal_wid = datas[0].petal_width
        for data in datas:
            max_sepal_len = max(max_sepal_len, data.sepal_length)
            max_sepal_wid = max(max_sepal_wid, data.sepal_width)
            max_petal_len = max(max_petal_len, data.petal_length)
            max_petal_wid = max(max_petal_wid, data.petal_width)
            min_sepal_len = min(min_sepal_len, data.sepal_length)
            min_sepal_wid = min(min_sepal_wid, data.sepal_width)
            min_petal_len = min(min_petal_len, data.petal_length)
            min_petal_wid = min(min_petal_wid, data.petal_width)
        if min_sepal_len < 0:
            min_sepal_len = -min_sepal_len
        self.scale_sepal_length = max(max_sepal_len, min_sepal_len)
        if min_sepal_wid < 0:
            min_sepal_wid = -min_sepal_wid
        self.scale_sepal_width = max(max_sepal_wid, min_sepal_wid)
        if min_petal_len < 0:
            min_petal_len = -min_petal_len
        self.scale_petal_length = max(max_petal_len, min_petal_len)
        if min_petal_wid < 0:
            min_petal_wid = -min_petal_wid
        self.scale_petal_width = max(max_petal_wid, min_petal_wid)
        # rescale datas
        for data in datas:
            data.sepal_length = data.sepal_length / self.scale_sepal_length * self.dev
            data.sepal_width = data.sepal_width / self.scale_sepal_width * self.dev
            data.petal_length = data.petal_length / self.scale_petal_length * self.dev
            data.petal_width = data.petal_width / self.scale_petal_width * self.dev

        return datas

    def make_folds(self, datas, training, validation, testing):
        # 60% training 20% testing 20% validation
        random.shuffle(datas)
        total = len(datas)
        training_index = int(total * 0.6)
        testing_index = int(total * 0.8)
        for data in range(0, training_index):
            training.append(datas[data])
        for data in range(training_index, testing_index):
            testing.append(datas[data])
        for data in range(testing_index, total):
            validation.append(datas[data])

    def preprocess(self, raw_data, training, validation, testing):
        datas = self.transformer(raw_data)
        self.decorrelate(datas)
        self.rescale(datas)
        setosas = []
        versicolours = []
        virginicas = []
        for data in datas:
            if data.setosa == 1:
                setosas.append(data)
            elif data.versicolour == 1:
                versicolours.append(data)
            elif data.virginica == 1:
                virginicas.append(data)

        self.make_folds(setosas, training, validation, testing)
        self.make_folds(versicolours, training, validation, testing)
        self.make_folds(virginicas, training, validation, testing)

        random.shuffle(training)
        random.shuffle(validation)
        random.shuffle(testing)

    def sigmoid(self, potential):
        return 1 / (1 + math.exp(-potential))

    def forward(self, sepal_length, sepal_width, petal_length, petal_width):
        self.outputs[self.INPUT_LAYER][self.SEPAL_LENGTH_INPUT] = sepal_length
        self.outputs[self.INPUT_LAYER][self.SEPAL_WIDTH_INPUT] = sepal_width
        self.outputs[self.INPUT_LAYER][self.PETAL_LENGTH_INPUT] = petal_length
        self.outputs[self.INPUT_LAYER][self.PETAL_WIDTH_INPUT] = petal_width

        # calculate potentials
        for layer in range(1, len(self.topology)):
            for neuron in range(0, self.topology[layer]):
                self.potentials[layer][neuron] = 0
                # w * B
                self.potentials[layer][neuron] += self.bias_weights[layer][neuron] * self.bias
                for parent in range(0, self.topology[layer - 1]):
                    self.potentials[layer][neuron] += self.weights[layer-1][parent][neuron] * self.outputs[layer - 1][parent]
                self.outputs[layer][neuron] = self.sigmoid(self.potentials[layer][neuron])

    def backward(self, setosa, versicolour, virginica):
        target = setosa
        output = self.outputs[self.OUTPUT_LAYER][self.SETOSA_OUTPUT]
        self.deltas[self.OUTPUT_LAYER][self.SETOSA_OUTPUT] = output * (1 - output) * (target - output)

        target = versicolour
        output = self.outputs[self.OUTPUT_LAYER][self.VERSICOLOUR_OUTPUT]
        self.deltas[self.OUTPUT_LAYER][self.VERSICOLOUR_OUTPUT] = output * (1 - output) * (target - output)

        target = virginica
        output = self.outputs[self.OUTPUT_LAYER][self.VIRGINICA_OUTPUT]
        self.deltas[self.OUTPUT_LAYER][self.VIRGINICA_OUTPUT] = output * (1 - output) * (target - output)

        # error
        for layer in range(self.OUTPUT_LAYER - 1, self.INPUT_LAYER, -1):
            for parent in range(0, self.topology[layer]):
                self.deltas[layer][parent] = 0
                for child in range(0, self.topology[layer + 1]):
                    self.deltas[layer][parent] += self.weights[layer][parent][child] * self.deltas[layer + 1][child]
                self.deltas[layer][parent] *= self.outputs[layer][parent] * (1 - self.outputs[layer][parent])

        # update weights
        for layer in range(self.OUTPUT_LAYER, self.INPUT_LAYER, -1):
            for child in range(0, self.topology[layer]):
                child_delta = self.deltas[layer][child]
                # bias
                self.bias_weights[layer][child] += self.learning_rate * self.bias * child_delta
                # neuron
                for parent in range(0, self.topology[layer - 1]):
                    self.weights[layer - 1][parent][child] += self.learning_rate * self.outputs[layer - 1][parent] * child_delta

    def validation(self, datas):
        mse_setosa = 0
        mse_versicolour = 0
        mse_virginica = 0
        for data in datas:
            self.forward(data.sepal_length, data.sepal_width, data.petal_length, data.petal_width)

            error = data.setosa - self.outputs[self.OUTPUT_LAYER][self.SETOSA_OUTPUT]
            mse_setosa += error ** 2

            error = data.versicolour - self.outputs[self.OUTPUT_LAYER][self.VERSICOLOUR_OUTPUT]
            mse_versicolour += error ** 2

            error = data.virginica - self.outputs[self.OUTPUT_LAYER][self.VIRGINICA_OUTPUT]
            mse_virginica += error ** 2

        total = len(datas)
        mse_setosa /= total
        mse_versicolour /= total
        mse_virginica /= total
        if VISIBLE == True:
            print("MSE of Setosa: " + str(mse_setosa))
            print("MSE of Versicolour: " + str(mse_versicolour))
            print("MSE of Virginica: " + str(mse_virginica))
        return mse_setosa + mse_versicolour + mse_virginica

    def get_result(self):
        setosa = self.outputs[self.OUTPUT_LAYER][self.SETOSA_OUTPUT]
        versicolour = self.outputs[self.OUTPUT_LAYER][self.VERSICOLOUR_OUTPUT]
        virginica = self.outputs[self.OUTPUT_LAYER][self.VIRGINICA_OUTPUT]
        max_output = max(setosa, versicolour, virginica)
        if setosa == max_output:
            return self.SETOSA_OUTPUT
        elif versicolour == max_output:
            return self.VERSICOLOUR_OUTPUT
        elif virginica == max_output:
            return self.VIRGINICA_OUTPUT


    def performance(self, datas):
        confusion = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        for data in datas:
            self.forward(data.sepal_length, data.sepal_width, data.petal_length, data.petal_width)

            if data.setosa == 1:
                confusion[self.SETOSA_OUTPUT][self.get_result()] += 1
            elif data.versicolour == 1:
                confusion[self.VERSICOLOUR_OUTPUT][self.get_result()] += 1
            elif data.virginica == 1:
                confusion[self.VIRGINICA_OUTPUT][self.get_result()] += 1

        accuracy = 0
        accuracy += confusion[self.SETOSA_OUTPUT][self.SETOSA_OUTPUT]
        accuracy += confusion[self.VERSICOLOUR_OUTPUT][self.VERSICOLOUR_OUTPUT]
        accuracy += confusion[self.VIRGINICA_OUTPUT][self.VIRGINICA_OUTPUT]
        accuracy /= len(datas)

        for row in range(0, len(confusion)):
            subtotal = 0
            for col in range(0, len(confusion[row])):
                subtotal += confusion[row][col]
            for col in range(0, len(confusion[row])):
                confusion[row][col] /= subtotal
                confusion[row][col] *= 100
                confusion[row][col] = int(confusion[row][col])

        print("counfusion matrix: ")
        print("\t\t\t\t predict Setosa|predict Versicolour|predict Virginica")
        print("truth Setosa\t\t", end="")
        for data in confusion[0]:
            print(data, "%", sep="", end="\t\t\t\t")
        print("")
        print("truth Versicolour\t\t", end="")
        for data in confusion[1]:
            print(data, "%", sep="", end="\t\t\t\t")
        print("")
        print("truth Virginica\t\t\t", end="")
        for data in confusion[2]:
            print(data, "%", sep="", end="\t\t\t\t")
        return accuracy

    def train(self, raw_data):
        training_set = list()
        validation_set = list()
        testing_set = []
        self.preprocess(raw_data, training_set, validation_set, testing_set)

        print("-----------Start Training-----------")
        prev_mse = 0
        if VISIBLE == False:
            print("training now, but training steps not be shown")
        for iter in range(0, 10000):
            if VISIBLE == True:
                print("iter", iter)
            for data in training_set:
                self.forward(data.sepal_length, data.sepal_width, data.petal_length, data.petal_width)
                self.backward(data.setosa, data.versicolour, data.virginica)
            curr_mse = self.validation(validation_set)
            if VISIBLE == True:
                print("MSE over all data: " + str(curr_mse))
                print()
            if abs(curr_mse - prev_mse) / curr_mse < 0.001:
                break
            prev_mse = curr_mse
        print("-----------End Training-----------")

        print("-----------Start Testing-----------")
        accuracy = self.performance(testing_set)
        print("\naccuracy over all = " + str(accuracy))
        print("-----------Start Training-----------")

    def classify(self, sepal_length, sepal_width, petal_length, petal_width):
        sepal_length = (sepal_length - self.ave_sepal_len) / self.var_sepal_length
        sepal_width = (sepal_width - self.ave_sepal_wid) / self.var_sepal_width
        petal_length = (petal_length - self.ave_petal_len) / self.var_petal_length
        petal_width = (petal_width - self.ave_petal_wid) / self.var_petal_width

        # rescale
        sepal_length = sepal_length / self.scale_sepal_length * self.dev
        sepal_width = sepal_width / self.scale_sepal_width * self.dev
        petal_length = petal_length / self.scale_petal_length * self.dev
        petal_width = petal_width / self.scale_petal_width * self.dev

        self.forward(sepal_length, sepal_width, petal_length, petal_width)

        result = self.get_result()

        if result == self.SETOSA_OUTPUT:
            return "Iris-setosa"
        elif result == self.VERSICOLOUR_OUTPUT:
            return "Iris-versicolor"
        elif result == self.VIRGINICA_OUTPUT:
            return "Iris-virginica"
        return "Unknown"

if __name__ == "__main__":
    print("---------------Important Information---------------")
    print("Max iteration in training step is 1000 and acceptable mse is 0.001, one of these conditions satisfied, training ends")
    print("Since we initiate weights in random fashion, every time you start you may get different accuracy; thus, there's"
          " no need to worry if you only get a low accuracy")
    print("On my computer, bias=1 and learning rate=0.1 works best(up to 100% overall accuracy)")
    print("To show or not to show training steps is up on you")
    print("Follow the instruction")
    print("---------------Tune Parameters---------------")
    BIAS = float(input("Input bias(positive float number): "))
    LEARNING_RATE - float(input("Input learning rate(positive float numebr): "))
    message = input("Do you want to show training details?[y/n]: ")
    if message == "y":
        VISIBLE = True
    elif message == "n":
        VISIBLE = False

    ann = ANN()
    ann.train("ANN - Iris data.txt")

    print("---------------Classify Manual Input Attributes---------------")
    while True:
        print("Follow the instruction, the unit of each attribute would be cm")
        print("If you want to end program, input negative values")
        sepal_length = float(input("Input sepal length in cm: "))
        if sepal_length < 0:
           break
        sepal_width = float(input("Input sepal width in cm: "))
        if sepal_width < 0:
            break
        petal_length = float(input("Input petal length in cm: "))
        if petal_length < 0:
            break
        petal_width = float(input("Input petal width in cm: "))
        if petal_width < 0:
            break
        print("predicted Iris label:", ann.classify(sepal_length, sepal_width, petal_length, petal_width))
