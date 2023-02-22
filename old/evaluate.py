import os
os.environ['PYTHONHASHSEED'] = '42'
from numpy.random import seed
from random import seed as rseed
from tensorflow.random import set_seed
seed(42)
rseed(42)
set_seed(42)
import models
from utils import *
from dataGenerator import *
from datasetProcess import *
from tensorflow.keras.optimizers import Adam
import argparse
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def evaluate(args):

    mode = args.mode # ["both","only_frames","only_differences"]

    if args.fusionType != 'C':
        if args.mode != 'both':
            print("Only Concat fusion supports one stream versions. Changing mode to /'both/'...")
            mode = "both"
        if args.lstmType == '3dconvblock':
            raise Exception('3dconvblock instead of lstm is only available for fusionType C ! aborting execution...')

    if args.fusionType == 'C':
        model_function = models.getProposedModelC
    elif args.fusionType == 'A':
        model_function = models.getProposedModelA
    elif args.fusionType == 'M':
        model_function = models.getProposedModelM

    dirinp = args.dirinp 

    dataset = args.dataset

    batch_size = args.batchSize

    vid_len = args.vidLen

    frame_diff_interval = 1

    input_frame_size = 224

    lstm_type = args.lstmType 

    itv = args.interval

    save_path = args.savePath


    #---------------------------------------------------
    if dataset == "rwf2000":
        dataset_frame_size = 320
    else:
        dataset_frame_size = 224

    preprocess_data = args.preprocessData

    weightsPath = args.weightsPath

    one_hot = False


    #----------------------------------------------------

    if preprocess_data:

        if not os.path.exists(os.path.join(dataset, 'processed')):
            os.makedirs(os.path.join(dataset, 'processed'))
        convert_dataset_to_npy_evl(src= dirinp, dest='{}/processed'.format(
            dataset), crop_x_y=None, target_frames=vid_len, frame_size= dataset_frame_size, interval=itv)
 
    test_generator = DataGenerator(directory = '{}/processed/test'.format(dataset),
                                    batch_size = batch_size,
                                    data_augmentation = False,
                                    shuffle = False,
                                    one_hot = one_hot,
                                    sample = False,
                                    resize = input_frame_size,
                                    background_suppress = True,
                                    target_frames = vid_len,
                                    dataset = dataset,
                                    mode = mode)
    #--------------------------------------------------

    print('> creating new model...')
    model = model_function(size=input_frame_size
                        , seq_len=vid_len
                        , frame_diff_interval = frame_diff_interval
                        , mode=mode
                        , lstm_type=lstm_type)

    optimizer = Adam(lr=4e-04, amsgrad=True)
    loss = 'binary_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc',f1_m,precision_m, recall_m])
    model.load_weights(f'{weightsPath}').expect_partial()
    model.trainable = False

    #--------------------------------------------------

    test_results = model.evaluate(
        steps = len(test_generator)
        , x = test_generator
        , verbose = 1
        , workers = 8
        , use_multiprocessing = False
    )
    
    print("====================")
    print("     Results        ")
    print("====================")
    print("> Test Loss:", test_results[0])
    print("> Test Accuracy:", test_results[1])
    print("> Test F1:", test_results[2])
    print("====================")
    save_as_csv(test_results, save_path, 'test_resuls.csv')

def setArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vidLen', type=int, default=32, help='Number of frames in a clip')
    parser.add_argument('--batchSize', type=int, default=4, help='Training batch size')
    parser.add_argument('--preprocessData', help='whether need to preprocess data ( make npy file from video clips )',action='store_false')
    parser.add_argument('--mode', type=str, default='both', help='model type - both, only_frames, only_differences', choices=['both', 'only_frames', 'only_differences']) 
    parser.add_argument('--dirinp', type=str)
    parser.add_argument('--lstmType', type=str, default='sepconv', help='lstm - conv, sepconv, asepconv, 3dconvblock(use 3dconvblock instead of lstm)', choices=['sepconv','asepconv', 'conv', '3dconvblock'])
    parser.add_argument('--fusionType', type=str, default='M', help='fusion type - A for add, M for multiply, C for concat', choices=['C','A','M']) 
    # parser.add_argument('--rwfPretrainedPath', type=str, default='NOT_SET', help='path to the weights pretrained on rwf dataset')
    parser.add_argument('--savePath', type=str, default='/gdrive/My Drive/THESIS/Data', help='folder path to save the models')
    parser.add_argument('--weightsPath', type=str, default='NOT_SET', help='path to the weights pretrained on rwf dataset')
    parser.add_argument('--dataset', type=str, default='rwf2000', help='dataset - rwf2000, hockey, movies')
    parser.add_argument('--interval', type=int, default=1, help='interval between frames')
   

    # args = parser.parse_args()
    # train(args)
    return parser

def evaluateTwoStreamSeparateConvLSTM(dirinp
                                , save_path
                                , weights_path
                                , dataset = 'rwf2000'
                                , interval = 5
                                , vid_len = 32
                                , batch_size = 4
                                , lstm_type = 'sepconv'
                                , fusion_type = 'M'):

    args = setArgs().parse_args([
        '--vidLen', str(vid_len),
        '--batchSize', str(batch_size),
        '--dirinp', dirinp,
        '--lstmType', lstm_type,
        '--fusionType', fusion_type,
        '--savePath', save_path,
        '--dataset', dataset,
        '--weightsPath', weights_path,
        '--interval', str(interval)
    ])
    evaluate(args)
