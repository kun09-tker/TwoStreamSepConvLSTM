from .models import getProposedModelC
import os
from .utils import *
from .dataGenerator import DataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler
from .args import setArgs
def train(args):

    mode = args.mode #['both', 'only_frames', 'only_differences']

    if args.fusionType != 'C':
        if args.mode != 'both':
            print("Only Concat fusion supports one stream versions. Changing mode to /'both/'...")
            mode = "both"
        if args.lstmType == '3dconvblock':
            raise Exception('3dconvblock instead of lstm is only available for fusionType C ! aborting execution...')

    if args.fusionType == 'C':
        model_function = getProposedModelC
    # elif args.fusionType == 'A':
    #     model_function = models.getProposedModelA
    elif args.fusionType == 'M':
        model_function = models.getProposedModelM
    
    initial_learning_rate = args.LearningRate
    resume_learning_rate = args.resumeLearningRate
    dirinp = args.dirinp
    vid_len = args.vidLen
    batch_size = args.batchSize
    input_heatmap_size = args.HeatMapSize
    epochs = args.numEpochs
    create_new_model = ( not args.resume )
    dataset = args.DatasetName
    save_path = args.savePath
    resume_path = args.resumePath
    type_part = args.typePart
    frame_diff_interval = 1

    if resume_path == "NOT_SET":
        currentModelPath =  os.path.join(save_path , str(dataset) + '_currentModel')
    else:
        currentModelPath = resume_path
    
    bestValPath =  os.path.join(save_path, str(dataset) + '_best_val_acc_Model')

    cnn_trainable = bool(args.cnnTrainable)
    loss = 'binary_crossentropy'

    train_generator = DataGenerator(directory = f'{dirinp}/train.pkl',
                                    type_part=type_part,
                                    batch_size = batch_size,
                                    shuffle = True,
                                    resize = input_heatmap_size,
                                    target_heatmap = vid_len,
                                    data_augmentation=True,
                                    mode = mode)
    
    val_generator = DataGenerator(directory = f'{dirinp}/val.pkl',
                                type_part=type_part,
                                batch_size = batch_size,
                                shuffle = False,
                                resize = input_heatmap_size,
                                target_heatmap = vid_len,
                                data_augmentation=False,
                                mode = mode)

    
    print('> cnn_trainable : ',cnn_trainable)
    if create_new_model:
        print('> creating new model...')
        model = model_function(size=input_heatmap_size, seq_len=vid_len,cnn_trainable=cnn_trainable, mode=mode, frame_diff_interval=frame_diff_interval)
        # if dataset == "hockey" or dataset == "movies":
        #     print('> loading weights pretrained on rwf dataset from', rwfPretrainedPath)
        #     model.load_weights(rwfPretrainedPath)
        optimizer = Adam(learning_rate=initial_learning_rate, amsgrad=True)
        model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
        print('> new model created')    
    else:
        print('> getting the model from...', currentModelPath)  
        # if dataset == 'rwf2000':
        model =  model_function(size=input_heatmap_size, seq_len=vid_len,cnn_trainable=cnn_trainable, mode=mode, frame_diff_interval=frame_diff_interval)
        optimizer = Adam(learning_rate=resume_learning_rate, amsgrad=True)
        model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
        model.load_weights(f'{currentModelPath}')
        # elif  dataset == "hockey" or dataset == "movies":
        #     model =  model_function(size=input_frame_size, seq_len=vid_len,cnn_trainable=cnn_trainable, frame_diff_interval = frame_diff_interval, mode=mode, lstm_type=lstm_type)
        #     optimizer = Adam(lr=initial_learning_rate, amsgrad=True)
        #     model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
        #     model.load_weights(f'{currentModelPath}')           

    print('> Summary of the model : ')
    model.summary(line_length=140)
    print('> Optimizer : ', model.optimizer.get_config())

    # dot_img_file = 'model_architecture.png'
    # print('> plotting the model architecture and saving at ', dot_img_file)
    # plot_model(model, to_file=dot_img_file, show_shapes=True)

    #--------------------------------------------------

    modelcheckpoint = ModelCheckpoint(
        currentModelPath, monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', save_freq='epoch')
        
    modelcheckpointVal = ModelCheckpoint(
        bestValPath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')

    historySavePath = os.path.join(save_path, 'results', str(dataset))
    save_training_history = SaveTrainingCurves(save_path = historySavePath)

    callback_list = [
                    modelcheckpoint,
                    modelcheckpointVal,
                    save_training_history
                    ]
                    
    callback_list.append(LearningRateScheduler(lr_scheduler, verbose = 0))
                    
    #--------------------------------------------------

    model.fit(
        steps_per_epoch=len(train_generator),
        x=train_generator,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        verbose=1,
        workers=8,
        max_queue_size=8,
        use_multiprocessing=False,
        callbacks= callback_list
    )

    # version = args.version
    
    # model.save(f'{dataset}_save_version_{version}')


def trainTwoStreamSeparateConvLSTM(dirinp
                                , save_path
                                , mode = 'both'
                                , dataset_name = 'rwf2000'
                                , resume_path = 'NOT_SET'
                                , resume = False
                                , resume_learning_rate = 1e-06
                                , num_epochs = 50
                                , vid_len = 32
                                , batch_size = 4
                                , heatmap_size = 224
                                , learning_rate = 1e-06
                                # , version = 0
                                , cnn_trainable = 1
                                , type_part = 'limb'
                                ):
    if resume:
        args = setArgs().parse_args([
            '--numEpochs', str(num_epochs),
            '--vidLen', str(vid_len),
            '--batchSize', str(batch_size),
            '--LearningRate', str(learning_rate),
            '--mode', mode,
            '--resume',
            '--dirinp', dirinp,
            '--savePath', save_path,
            '--resumePath', resume_path,
            '--resumeLearningRate', str(resume_learning_rate),
            '--DatasetName', dataset_name,
            '--HeatMapSize', str(heatmap_size),
            '--cnnTrainable', str(cnn_trainable),
            '--typePart', type_part
        ])
    else:
        args = setArgs().parse_args([
            '--numEpochs', str(num_epochs),
            '--vidLen', str(vid_len),
            '--batchSize', str(batch_size),
            '--LearningRate', str(learning_rate),
            '--mode', mode,
            '--dirinp', dirinp,
            '--savePath', save_path,
            '--DatasetName', dataset_name,
            '--HeatMapSize', str(heatmap_size),
            '--cnnTrainable', str(cnn_trainable),
            '--typePart', type_part
        ])
    train(args)