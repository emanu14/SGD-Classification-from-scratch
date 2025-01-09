import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler



# ONE-HOT ENCODING
def labels_to_onehot(y_raw, nlabels):
    y = np.zeros((y_raw.shape[0], nlabels))
    for i in range(y.shape[0]):
        y[i][int(y_raw[i][0]-1)] = 1
    return y



# LABEL ENCODING
def onehot_to_label(y):
    # y is 2-dimensional
    return np.argmax(y, axis=1)




# LOAD DERMATOLOGY DATASET
def get_data():
    inputs = 34
    outputs = 6

    data_array = []

    file_path = 'dermatology_classification/dermatology.dat'

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            sample = list(map(float, line.split(' ')))
            data_array.append(sample)

    data = np.array(data_array)

    x = data[:, :inputs]
    x = np.hstack((x, np.ones((x.shape[0], 1))))

    y = labels_to_onehot(data[:, -1:], outputs)

    return x, y




# FUNCIÓN DE INICIALIZACIÓN DE PARÁMETROS (XAVIER)
def params_init(inputs, outputs):
    std_dev = np.sqrt(2 / (inputs + outputs))
    params = np.random.normal(0, std_dev, size=(inputs, outputs))
    return params



# FUNCIÓN DE ETIQUETADO DE PROBABILIDADES
def probs_to_binary(x, threshold=0.5):
    # x is 2-dimensional
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j] > threshold:
                x[i][j] = 1
            else:
                x[i][j] = 0
    
    return x




# FUNCIÓN DE HIPÓTESIS
def predict(x, params):
    return np.dot(x, params)




# FUNCIÓN DE PROBABILIDAD (SOFTMAX)
def softmax(x, params):
    exp = np.exp(predict(x, params))
    return exp / np.sum(exp, axis=1, keepdims=True)




# FUNCIÓN DE COSTO (LOG-VEROSIMILITUD)
def logvn(x, y, params, lambda_l1):
    s = softmax(x, params)
    return -np.sum(y * np.log(s)) + lambda_l1 * np.sum(np.abs(params))




# FUNCIÓN GRADIENTE (DE LA FUNCIÓN DE COSTO)
def gradient_MLF(x, y, params, lambda_l1):
    # using the normalized probabilities
    s = softmax(x, params)
    return np.dot(x.T, (s - y)) + lambda_l1 * np.sign(params)




# STOCHASTIC GRADIENT DESCENT
def trainSGD(
        params, 
        lr,
        lambda_l1=0,
        momentum=0, 
        dampening=0, 
        weight_decay=0, 
        nesterov=False, 
        maximize=False,
    ):
    w = params
    t = 0
    epochs = 300

    lvn = logvn(X_train, y_train, w, lambda_l1)

    while (t < epochs):

        g = gradient_MLF(X_train, y_train, w, lambda_l1)

        if (weight_decay != 0):
            g = (g + (weight_decay * w))
        
        if (momentum != 0):

            if t > 0:
                b = (momentum * b) + ((1 - dampening) * g)
            else:
                b = g
            
            if nesterov:
                g = g + (momentum * b)
            else:
                g = b
        
        if maximize:
            w = (w + (lr * g))
        else:
            w = (w - (lr * g))
        

        lvn = logvn(X_train, y_train, w, lambda_l1)

        if t % 1 == 0:
            print(f'<logvn = {lvn}>')
        
        t += 1

    return w




def plot_confusion_matrix(y_test, y_pred):
    # y_test and y_pred should be on label encoding format (as opposed to one-hot)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', xticklabels=['1', '2', '3', '4', '5', '6'], yticklabels=['1', '2', '3', '4', '5', '6'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    
    

def train_test(X, y, params_init):

    # preprocesamiento: train-test split
    # por lotes
    global X_train, y_train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    optimized_params = trainSGD(
        params = params_init,
        lr = 0.0005,
        lambda_l1 = 0.1,
        momentum = 0.3,
        dampening = 0.01,
    )

    y_pred_probs = softmax(X_test, optimized_params)
    y_pred = probs_to_binary(y_pred_probs)

    # print(f'\ny prediction\n{y_pred}')

    plot_confusion_matrix(
        onehot_to_label(y_test), 
        onehot_to_label(y_pred),
    )





X, y = get_data()
train_test(X, y, params_init(len(X[0]), len(y[0])))