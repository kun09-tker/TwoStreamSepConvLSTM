from .models import getProposedModelC
import os
from .utils import *
from .dataGenerator import DataGenerator
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import backend as K
from .args import setArgs

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



def evaluate(args, data_test):

    mode = args.mode #['both', 'only_frames', 'only_differences']
    backbone = args.backBone

    # if args.fusionType != 'C':
    #     if args.mode != 'both':
    #         print("Only Concat fusion supports one stream versions. Changing mode to /'both/'...")
    #         mode = "both"
    #     if args.lstmType == '3dconvblock':
    #         raise Exception('3dconvblock instead of lstm is only available for fusionType C ! aborting execution...')

    if args.fusionType == 'C':
        model_function = getProposedModelC
    # elif args.fusionType == 'A':
    #     model_function = models.getProposedModelA
    # elif args.fusionType == 'M':
    #     model_function = getProposedModelM
    
    initial_learning_rate = args.LearningRate
    dirinp = args.dirinp
    vid_len = args.vidLen
    batch_size = args.batchSize
    input_heatmap_size = args.HeatMapSize
    dataset = args.DatasetName
    save_path = args.savePath
    resume_path = args.resumePath
    type_part = args.typePart
    frame_diff_interval = 1

    if resume_path == "NOT_SET":
        currentModelPath =  os.path.join(save_path , str(dataset) + '_currentModel')
    else:
        currentModelPath = resume_path

    # cnn_trainable = bool(args.cnnTrainable)
    loss = 'binary_crossentropy'

    test_generator = DataGenerator(directory = f'{dirinp}/{data_test}.pkl',
                                sample = False,
                                type_part=type_part,
                                batch_size = batch_size,
                                shuffle = False,
                                resize = input_heatmap_size,
                                target_heatmap = vid_len,
                                data_augmentation=False,
                                mode = mode)

    print('> getting the model from...', resume_path) 
    model = model_function(size=input_heatmap_size, seq_len=vid_len,cnn_trainable=False, mode=mode, frame_diff_interval=frame_diff_interval, backbone=backbone)
    optimizer = Adam(learning_rate=initial_learning_rate, amsgrad=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["acc", f1_m]) 
    model.load_weights(f'{currentModelPath}')
    model.trainable = False
        

    print('> Summary of the model : ')
    model.summary(line_length=140)
    print('> Optimizer : ', model.optimizer.get_config())

    test_results = model.evaluate(
        steps = len(test_generator),
        x=test_generator,
        verbose=1,
        workers=8,
        max_queue_size=8,
        use_multiprocessing=False,
    )
    print("====================")
    print("     Results        ")
    print("====================")
    print("> Test Loss:", test_results[0])
    print("> Test acc_score:", test_results[1])
    print("> Test precision_score:", test_results[2])
    print("> Test recall_score:", test_results[3])
    print("> Test f1_score:", test_results[4])
    print("====================")
    # save_as_csv(train_results, "", 'train_results.csv')
    # save_as_csv(test_results, "", 'test_resuls.csv')




def evaluateTwoStreamSeparateConvLSTM(dirinp
                                , save_path
                                , mode = 'both'
                                , dataset_name = 'rwf2000'
                                , fusion_type = 'C'
                                , resume_path = 'NOT_SET'
                                , resume = False
                                , resume_learning_rate = 1e-06
                                , num_epochs = 50
                                , vid_len = 32
                                , batch_size = 4
                                , heatmap_size = 224
                                , learning_rate = 1e-06
                                # , version = 0
                                , data_test = 'test'
                                , cnn_trainable = 1
                                , type_part = 'limb'
                                , backbone = 'mobilenetv2'
                                ):
    if resume:
        args = setArgs().parse_args([
            '--numEpochs', str(num_epochs),
            '--vidLen', str(vid_len),
            '--batchSize', str(batch_size),
            '--LearningRate', str(learning_rate),
            '--fusionType', fusion_type,
            '--mode', mode,
            '--resume',
            '--dirinp', dirinp,
            '--savePath', save_path,
            '--resumePath', resume_path,
            '--resumeLearningRate', str(resume_learning_rate),
            '--DatasetName', dataset_name,
            '--HeatMapSize', str(heatmap_size),
            '--cnnTrainable', str(cnn_trainable),
            '--typePart', type_part,
            '--backBone', backbone
        ])
    else:
        args = setArgs().parse_args([
            '--numEpochs', str(num_epochs),
            '--vidLen', str(vid_len),
            '--batchSize', str(batch_size),
            '--LearningRate', str(learning_rate),
            '--fusionType', fusion_type,
            '--mode', mode,
            '--dirinp', dirinp,
            '--savePath', save_path,
            '--DatasetName', dataset_name,
            '--HeatMapSize', str(heatmap_size),
            '--cnnTrainable', str(cnn_trainable),
            '--typePart', type_part,
            '--backBone', backbone
        ])
    evaluate(args, data_test)