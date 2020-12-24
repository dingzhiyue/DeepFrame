import numpy as np

def accuracy(y_pred:'tensor', y_true:'tensor', threshold:'float'=0.5)->'scale':
    '''
    binay now...to do: support softmax
    '''
    prediction = np.zeros_like(y_pred.data)
    prediction[y_pred.data>=threshold] = 1
    correct = prediction[prediction==y_true.data]
    return correct.shape[0]/prediction.shape[0]
