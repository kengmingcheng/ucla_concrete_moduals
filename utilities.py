import os
import sys
import matplotlib as plt
import math
from tensorflow.keras import backend as K

def plot_history(history, label, save_path="./img/"):
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss for %s' % label)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    _= plt.legend(['Train','Validation'], loc='upper left')
    plt.savefig(os.path.join(save_path, label + '_Loss.png'), format='png')
    plt.show()

def measure_v_predict(train_target_inverse, train_predict_inverse, cv_target_inverse, cv_predict_inverse, title='test', save_path="./img/"):
    plt.plot(train_target_inverse, train_predict_inverse, marker='o', linestyle='None')
    plt.plot(cv_target_inverse, cv_predict_inverse, marker='x', linestyle='None')
    plt.plot([0,100],[0,100],'#303030', ls='--')
    plt.xlabel('measured value')
    plt.ylabel('predicted value')
    _= plt.legend(['Train','Validation'], loc='upper left')
    plt.savefig(os.path.join(save_path, title+'.png'), format='png')
    plt.show()

def std(nums):
    avg = sum(nums)/len(nums)
    dev = 0
    for num in nums:
        dev += (num - avg)**2
    return math.sqrt(dev/(len(nums)-1)) if len(nums) > 1 else 0

# keras functions
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )