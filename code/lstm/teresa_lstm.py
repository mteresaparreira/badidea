#runs 50 seeds for every dataframe. gets agent and inferences.

#imports

import argparse


from sklearn.linear_model import SGDClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors, datasets
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state
from copy import deepcopy
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
import pickle
from keras.models import load_model

#import tensorflow as tf
#from tensorflow. import keras

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#import imblearn
import collections

import tensorflow as tf
import tensorflow_addons as tfa
import sklearn
from sklearn.metrics import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report,confusion_matrix

#from tensorflow.keras.models import Sequential
from keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import concatenate
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Softmax
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import random
from collections import Counter
#from imblearn.under_sampling import OneSidedSelection
#from imblearn.over_sampling import ADASYN
import pickle


#seeded!!!!!!!!!!!!!!
#seed_value = 42

#random.seed(seed_value)
#from numpy.random import seed
#seed(seed_value)
#from tensorflow import set_random_seed
#tf.random.set_seed(seed_value)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout





from tensorflow import keras
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2., alpha=4.,
                 reduction=keras.losses.Reduction.AUTO, name='focal_loss'):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})
        """
        super(FocalLoss, self).__init__(reduction=reduction,
                                        name=name)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def call(self, y_true, y_pred):
        """
        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(
            tf.subtract(1., model_out), self.gamma))
        fl = tf.multiply(self.alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)





def create_dataset(data_x, look_back=1):
    data_x = np.nan_to_num(data_x)
    dataX  = []
    for i in range(len(data_x)-look_back):
        a = data_x[i:(i+look_back)]
        dataX.append(a)
        #dataY.append(dataset[i + look_back, 0])
    return np.array(dataX) #, numpy.array(dataY)


def classification_report_tolerance(y_pred,y_true,margin = 1):
    
    metrics_dict = dict()
    all_metrics_dict = dict()
    classes = np.unique(y_true)
    all_p = []
    all_r = []
    all_a = []
    all_f1 = []
    
    for classi in classes:
        tp = 0
        tn = 0
        fp = 0
        fn = 0


        for i, y in enumerate(y_pred):
            if y == classi:
                if y in y_true[i-margin:i+margin+1]:
                    #print(y_true[i-1:i+1].shape)
                    tp = tp + 1
                else:
                    fp = fp + 1
            else:
                if y not in y_true[i-1:i+1]:
                    fn = fn + 1
                else:
                    tn = tn + 1
        precision = tp/(tp+fp)
        recall = tp / (tp + fn)
        f1 = (2*precision*recall)/(precision+recall)
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        metrics_dict[classi] = [tp, tn, fp, fp]
        all_metrics_dict[classi] = [precision, recall, f1, accuracy]
        all_p.append(precision)
        all_r.append(recall)
        all_a.append(accuracy)
        all_f1.append(f1)
        
    
    print(metrics_dict)
    print(all_metrics_dict)
    
    macro_dict = dict()
    macro_dict['macro-precision'] = sum(np.array(all_p))/len(all_p)
    macro_dict['macro-recall'] = sum(np.array(all_r))/len(all_r)
    macro_dict['macro-accuracy'] = sum(np.array(all_a))/len(all_a)
    macro_dict['macro-f1'] = (2* macro_dict['macro-precision']*macro_dict['macro-recall'])/(macro_dict['macro-recall'] + macro_dict['macro-precision'])
    
    print(macro_dict)
    
    return metrics_dict, all_metrics_dict, macro_dict



def get_split_data(df, seed = 0, mode = 'time_based', lookback = 1):
    # mode that we can define our train test set on
    # choose between 'episode_based', which will separate episodes to different sets,
    # or 'time_based' which will cut either the beginning or end bit of each interaction for training and testing
    #mode = 'time_based'#lookback = 30
    #lookback = 10
    data_x = {}
    y_type = {}
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if mode == 'episode_based':
        #np.count_nonzero(np.isnan(df.loc[:,'state_feature_0':state_names[-1]].values))
        test_set = []
        all_episodes = list(df['episode_name'].unique())
        for i in range(math.ceil(len(all_episodes)*0.2)):
            test_set.append(np.random.choice(all_episodes))
            all_episodes.remove(test_set[-1])
        random.shuffle(all_episodes)
        validation_set = []
        for i in range(math.ceil(len(all_episodes)*0.2)):
            validation_set.append(np.random.choice(all_episodes))
            all_episodes.remove(validation_set[-1])
        df_actions = df

        df_actions_val = df_actions[df_actions['episode_name'].map(lambda x: x in validation_set)]
        df_actions_test = df_actions[df_actions['episode_name'].map(lambda x: x in test_set)]
        df_actions = df_actions[df_actions['episode_name'].map(lambda x: x not in validation_set and x not in test_set)]
        datasets = {'train': df_actions, 'test': df_actions_test, 'val': df_actions_val}

        print(datasets['train'].shape, datasets['test'].shape, datasets['val'].shape)
        for key in datasets.keys():
            df_sel = datasets[key]
            X = create_dataset(df_sel.loc[:,'state_feature_0':state_names[-1]].values, lookback)
            #print(np.count_nonzero(np.isnan(X)))
            mask = df_sel['action_type']!=0
            data_x[key] = X#[mask[lookback:]]
            y_type[key] = df_sel.loc[:,'action_type'].values[lookback:][:] #[lookback:]#[mask[lookback:]]
            
            print(np.unique(y_type[key], return_counts=True))
            #y_type[key] = y_type[key]-1
            #print(np.unique(y_type[key], return_counts=True))
    elif mode == 'time_based':
        df_actions = df
        all_episodes = list(df['episode_name'].unique())
        for episode in all_episodes:
            data_episode = df_actions[df_actions['episode_name'] == episode]
            X = create_dataset(data_episode.loc[:,'state_feature_0':state_names[-1]].values, lookback)
            mask = data_episode['action_type']!=0 # gives a vector of True and False
            data_x_temp = X
            y_type_temp = data_episode.loc[:,'action_type'].values[lookback:][:] 

            # split test train val episode based (end of interaction gets cut off)
            number_examples = len(y_type_temp)
            #print(number_examples)
            start_index_test = math.ceil(number_examples*0.8)
            start_index_val = math.ceil(start_index_test*0.8)
            if number_examples == 0: 
                continue
            if 'train' not in data_x:
                data_x['train'] = data_x_temp[:start_index_val,:]
                data_x['val'] = data_x_temp[start_index_val:start_index_test,:]
                data_x['test'] = data_x_temp[start_index_test:,:]
                y_type['train'] = y_type_temp[:start_index_val]
                y_type['val'] = y_type_temp[start_index_val:start_index_test]
                y_type['test'] = y_type_temp[start_index_test:]

            else:
                data_x['train'] = np.vstack((data_x['train'], data_x_temp[:start_index_val,:]))
                data_x['val'] =  np.vstack((data_x['val'], data_x_temp[start_index_val:start_index_test,:]))
                data_x['test'] = np.vstack((data_x['test'], data_x_temp[start_index_test:,:]))
                y_type['train'] = np.hstack((y_type['train'], y_type_temp[:start_index_val]))
                y_type['val'] = np.hstack((y_type['val'], y_type_temp[start_index_val:start_index_test]))
                y_type['test'] = np.hstack((y_type['test'], y_type_temp[start_index_test:]))

    for key in data_x.keys():
        print(key, data_x[key].shape, y_type[key].shape, np.unique(y_type[key], return_counts=True))
    return data_x, y_type
    

def doExperiment(seed, mode = 'episode_based', lookbacks = [5,10,15],losses = ['focal', 'mean_squared_error','binary_crossentropy','hinge'],
                    optimizers = ['SGD', 'Adam' ], activations = ['sigmoid', 'relu', 'tanh', 'softmax'],dense_size = [25],epochs = 100, 
                    batch_sizes = [8,16,32,64], dropouts = [0,0.2,0.4,0.6] note = None ):


    database = '../ccdb/'
    df = pd.read_csv(database + 'df.csv')
    from numpy.random import seed
   
        
    # Directory 
    import datetime
    now = datetime.datetime.now()
    dt_string_folder = now.strftime("%Y_%m_%d_%H_%M_%S")
    import os 
    folder_name = dt_string_folder+'_' + note+'_'+mode
    os.mkdir('./results/' + folder_name) 

    seed = seed_value
    modelcount = 0

    for lookback in lookbacks:
        for loss in losses:
            for optimizer in optimizers:
                for activation in activations:
                    for batch_size in batch_sizes:
                        
                        try:

                            random.seed(seed_value)
                            seed(seed_value)
                            tf.random.set_seed(seed_value)
                        
                            #print(num_classes_type)#, num_classes_target)



                            #inputA = Input(shape=(lookback,len(state_names),), name="inputP1")

                            #inputsP = [inputA, inputB]#, inputC]
                            #models = []
                            #for elem in inputsP:
                            #    x = LSTM(18, return_sequences=True)(elem)
                            #    x = Model(inputs=elem, outputs=x)
                            #    models.append(x)
                            # combine the output of the three branches plus extra input
                            #combined = concatenate([models[0].output, models[1].output, models[2].output, inputD])
                            ###########
                            
                            data_x, y_type = get_split_data(df, seed = seed_value, mode = mode, lookback = lookback)

                            num_classes_type = len(np.unique(y_type['train']))
                            inputA = Input(shape=(lookback,len(states)), name="inputP1")
                            "inputB = Input(shape=(lookback,1,), name="inputP2")
                                #inputC = Input(shape=(lookback,len(states)-14,), name="inputP3")


                                #combined = concatenate([inputA,inputB, inputC])

                                #z = LSTM(2*lookback, return_sequences=True)(combined)
                                #z1 = LSTM(lookback, return_sequences=True)(inputA)
                                #z2 = LSTM(lookback, return_sequences=True)(combined)
                                #combinedz = concatenate([z1,z2])
                            
                            z = LSTM(lookback)(inputA)
                            #z1 = LSTM(lookback, return_sequences=True)(inputA)
                            #z2 = LSTM(lookback, return_sequences=True)(inputB)
                            #combinedz = concatenate([z1,z2])
                            #z = LSTM(lookback)(combinedz)
                            zd = Dropout(dropout)(z)
                            output_type = Dense(num_classes_type, activation = activation)(zd)
                            #output_type = Dense(num_classes_type, activation = activation)(z)
                            #model = Model(inputs=[inputA], outputs=output_type)
                            
                            #model = Model(inputs=[inputA,inputB], outputs=output_type)
                            model = Model(inputs=[inputA], outputs=output_type)
                            

                                #z = LSTM(26)(z)
                                #output_type = Dense(num_classes_type)(output_type)
                                #output_type = Dense(num_classes_type)(z)
                            #output_type = Dense(num_classes_type, activation = activation)(z)
                            #model = Model(inputs=[inputA], outputs=output_type)
                                #output_type = Softmax()(output_type)
                                #model =  tf.keras.Sequential()
                                #model.add(LSTM(lookback,return_sequences = True)(combined))
                                #model.add(Dense(25, activation='relu'))
                                #model.add(Dense(1, activation=activation))
                                #model = Model(inputs=inputs, outputs=outputs)
                            if loss == 'focal':
                                model.compile(loss=FocalLoss(alpha=1.3), optimizer=optimizer)#, metrics=['accuracy'])
                            else:
                                model.compile(loss=loss, optimizer=optimizer)#, metrics=['accuracy'])
                                # Compile model
                                #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                            

                            #combined = concatenate([inputB, inputC])

                            # create model
                            #model = KerasClassifier(build_fn=create_model(), epochs=100, verbose=2)
                            # define the grid search parameters

                            #model = Model(inputs=[inputA, inputB,inputC], outputs=output_type)
                            model.summary()
                            #model.compile(loss=
                            #              [FocalLoss(alpha=1.3), 
                            #               keras.losses.CategoricalCrossentropy(from_logits=True)
                            #              ], 
                            #              loss_weights=[0.3, 1.0],
                            #              optimizer='adam')

                            #print(model.build_fn().summary())

                            #model.compile(loss='mean_squared_error', optimizer='adam')


                            inputP1 = data_x['train'][:,:,:]
                            #inputP1 = data_x['train'][:,:,:34]
                            #inputP2 = data_x['train'][:,:,34:]
                            #inputP3 = data_x['train'][:,:,14:]
                            #inputG = data_x['train'][:,:,36:]
                            inputP1_val = data_x['val'][:,:,:]
                            #inputP1_val = data_x['val'][:,:,:34]
                            #inputP2_val = data_x['val'][:,:,34:]
                            #inputP3_val = data_x['val'][:,:,14:]
                            #inputP2_val = data_x['val'][:,:,12:24]
                            #inputP3_val = data_x['val'][:,:,24:36]
                            #inputG_val = data_x['val'][:,:,36:]
                            inputP1_test = data_x['test'][:,:,:]
                            #inputP1_test = data_x['test'][:,:,:34]
                            #inputP2_test = data_x['test'][:,:,34:]
                            #inputP3_test = data_x['test'][:,:,14:]
                            #inputP2_test = data_x['test'][:,:,12:24]
                            #inputP3_test = data_x['test'][:,:,24:36]
                            #inputG_test = data_x['test'][:,:,36:]
                            ###############
                            #print(inputP1.shape,inputP2.shape,inputP3.shape,#inputG.shape,
                            #      inputP1_val.shape,inputP2_val.shape,inputP3_val.shape,#inputG_val.shape,
                            #      inputP1_test.shape,inputP2_test.shape,inputP3_test.shape)#inputG_test.shape)

                            y_train_onehot_type = to_categorical(y_type["train"])
                            y_val_onehot_type = to_categorical(y_type["val"])
                            print(y_train_onehot_type.shape, y_val_onehot_type.shape)
                            #print( {"inputP1": inputP1, "inputP2": inputP2, "inputP3": inputP3, "inputG": inputG})

                            hist = model.fit(
                                #{"all": X_train_all},
                                {"inputP1": inputP1},#, "inputP2": inputP2, "inputP3": inputP3},#, "inputG": inputG},
                                #[y_train_onehot_type, y_train_onehot_target],
                                y_train_onehot_type,
                                epochs=epochs,
                                batch_size=batch_size,
                                verbose=2,
                                
                                validation_data=([inputP1_val], y_val_onehot_type)#, inputP2_val, inputP3_val, inputG_val], y_val_onehot_type)
                            )

                            y_pred_onehot_type  = model.predict({"inputP1": inputP1_test}),
                                                                #"inputP2": inputP2_test})
                                                                    #"inputP3": inputP3_test})#, "inputP2": inputP2_test, "inputP3": inputP3_test, "inputG": inputG_test})



                            y_pred_type = np.argmax(y_pred_onehot_type, axis=1)
                            #y_pred_target = np.argmax(y_pred_onehot_target, axis=1)
                            print("--------- Type ---------")
                            print(accuracy_score(y_type["test"], y_pred_type))
                            print(classification_report(y_type["test"], y_pred_type))
                            #print("--------- Target ---------")
                            #print(accuracy_score(y_target["test"], y_pred_target))
                            #print(classification_report(y_target["test"], y_pred_target))
                            print(hist.history['loss'][-1])
                            
                            d1,d2,d3 = classification_report_tolerance(y_type["test"], y_pred_type, 1)



                            label_type, count_type = np.unique(y_type["test"], return_counts=True)
                            label_pred_type, count_pred_type = np.unique(y_pred_type, return_counts=True)
                            df_type = pd.DataFrame(columns=label_type)
                            print(df_type)
                            df_type.loc["TestType"] = count_type
                            df_type.loc["PredType"] = np.zeros(len(count_type))
                            for i in range(len(label_pred_type)):
                                df_type.loc["PredType", label_pred_type[i]] = count_pred_type[i] 
                            df_type.loc["Percent"] = df_type.loc["PredType"].values/count_type
                            print(df_type)
                            now = datetime.datetime.now()
                            # dd/mm/YY H:M:S
                            dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")

                            print(dt_string)
                            import os
                            
                            print('seed ' + str(seed_value) + ' lookback' + str(lookback)+
                                        ' loss' + loss + ' optimizer' + optimizer + ' activation'+ 
                                        activation + ' batch_size' + str(batch_size))

                            with open('./results/' + folder_name + '/results'+'.txt', 'a') as f:
                                f.write(dt_string+"\n")
                                f.write('seed ' + str(seed_value) + ' lookback' + str(lookback)+
                                        ' loss' + loss + ' optimizer' + optimizer + ' activation'+ 
                                        activation + ' batch_size' + str(batch_size) + '\n')
                                

                                f.write("modelcount\t")
                                f.write(str(modelcount)+"\n")
                                f.write("states\t")
                                f.write(str(states)+"\n")
                                f.write("seed\t")
                                f.write(str(seed_value)+"\n")
                                f.write("mode\t")
                                f.write(mode+"\n")
                                f.write("dropout\t")
                                f.write(str(dropout)+"\n")
                                f.write("epochs\t")
                                f.write(str(epochs)+"\n")
                                f.write("batch_size\t")
                                f.write(str(batch_size)+"\n")
                                f.write("model summary\t")
                                #f.write(model.summary()+"\n")
                                model.summary(print_fn=lambda x: f.write(x + '\n'))
                                f.write("loss function\t")
                                f.write(str(loss)+"\n")
                                f.write("optimizer\t")
                                f.write(str(optimizer)+"\n")
                                f.write("loss results\t")
                                f.write(str(hist.history['loss'])+"\n")
                                f.write("val loss results\t")
                                f.write(str(hist.history['val_loss'])+"\n")
                                #f.write("Input shape\t")
                                #f.write(str(data_x['train'].shape)+"\n")
                                #f.write("Loss\t")
                                #f.write(str(hist.history['loss'][-1])+"\n")

                                f.write(df_type.to_string()+"\n")
                                #f.write(df_target.to_string())

                                f.write("----- Type -----\t")
                                f.write(classification_report(y_type["test"], y_pred_type)+"\n")
                                f.write("margin of error\t")
                                f.write(str(d1)+"\n")
                                f.write(str(d2)+"\n")
                                f.write(str(d3)+"\n")
                                
                                
                                #f.write("----- Target -----")
                                #f.write(classification_report(y_target["test"], y_pred_target))
                                f.write('\n-------------------------------------------\n\n')
                                
                                filename = str('test_4_may.h5'
                                #pickle.dump(model, open(filename, 'wb'))
                                model.save(filename)

                                modelcount = modelcount + 1

                        
                        except IndexError as error: 
                            print(error)
                            pass

                        except FileNotFoundError as error: 
                            print(error)
                            pass
                        except ValueError as error: 
                            print(error)
                            pass


    







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', metavar='seed', type=int, nargs='+',
                        help='The seed given to coach rl for training.')
    parser.add_argument('lookbacks', metavar='lookbacks', type=list, nargs='+',
                        help='The lookback given to coach rl for training.')
    parser.add_argument('mode', metavar='mode', type=str, nargs='+',
                        help='The mode given to coach rl for training.')
    parser.add_argument('losses', metavar='losses', type=list, nargs='+',
                        help='The loss given to coach rl for training.')
    parser.add_argument('optimizers', metavar='optimizers', type=list, nargs='+',
                        help='The optimizer given to coach rl for training.')
    parser.add_argument('activations', metavar='activations', type=list, nargs='+',
                        help='The act function given to coach rl for training.')   
    parser.add_argument('dense_size', metavar='dense_size', type=list, nargs='+',
                        help='The dense units given to coach rl for training.')     
    parser.add_argument('epochs', metavar='epochs', type=int, nargs='+',
                        help='The epochs given to coach rl for training.')
    parser.add_argument('batch_sizes', metavar='batch_sizes', type=list, nargs='+',
                        help='The batch size given to coach rl for training.')    
    parser.add_argument('dropouts', metavar='dropouts', type=list, nargs='+',
                        help='The dropout given to coach rl for training.')
    parser.add_argument('note', metavar='note', type=str, nargs='+',
                        help='A note')                                                                                    
    args = parser.parse_args()
    print(args.seeds)
    doExperiment(int(args.seed),str(args.mode), list(args.losses),list(args.optimizers),list(args.activations),
                list(args.dense_size),list(args.batch_sizes),list(args.dropouts), int(args.epochs), str(args.note))
