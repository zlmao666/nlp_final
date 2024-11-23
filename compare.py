import numpy as np

def get_accuracy(true, predicted):
    '''
    tp + tn / tp + fp + tn + fn   
    '''
    
    tp_tn = sum(np.logical_not(np.logical_xor(predicted, true)))
    tp_fp = sum(predicted)
    tn_fn = len(predicted) - tp_fp
    tn_fn1 = sum(np.logical_not(predicted))
    assert tn_fn == tn_fn1
    denom = tp_fp + tn_fn
    
    return tp_tn/denom


def get_precision(true, predicted):
    '''
    tp / tp + fp
    '''
    
    tp = np.dot(predicted, true)
    tp_fp = sum(predicted)
    
    return tp/tp_fp


def get_recall(true, predicted):
    '''
    tp / tp + fn   
    '''

    tp = np.dot(predicted, true)
    tp_fn = sum(true)
    
    return tp/tp_fn


def get_f1(true, predicted):
    '''Computes the F1 score as the harmonic mean of precision and recall.'''
    precision = get_precision(true, predicted)
    recall = get_recall(true, predicted)
    
    if precision + recall == 0:
        return 0
        
    return 2 * (precision * recall) / (precision + recall)
    
    
def get_metrics(true, predicted):
    accuracy = get_accuracy(true, predicted)
    precision = get_precision(true, predicted)
    recall = get_recall(true, predicted)
    f1 = get_f1(true, predicted)
    return accuracy, precision, recall, f1

def main():
    print("Run test_data.py to test data or final.py to play around with algorithm")

if __name__ == "__main__":
    main()