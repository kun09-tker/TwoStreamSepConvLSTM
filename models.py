from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Flatten, Dropout, ConvLSTM2D, BatchNormalization, Activation
from tensorflow.keras.layers import TimeDistributed, Multiply, Add
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import LeakyReLU, Conv3D, Multiply, MaxPooling3D, MaxPooling2D, Concatenate, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from .sep_conv_rnn import SepConvLSTM2D

def getProposedModelC(size=224, seq_len=32 , cnn_weight = 'imagenet',cnn_trainable = True, lstm_type='sepconv', weight_decay = 2e-5, frame_diff_interval = 1, mode = "both", cnn_dropout = 0.25, lstm_dropout = 0.25, dense_dropout = 0.3, seed = 42):
    """parameters:
    size = height/width of each frame,
    seq_len = number of frames in each sequence,
    cnn_weight= None or 'imagenet'
    mode = "only_frames" or "only_differences" or "both"
       returns:
    model
    """
    lstm_type='sepconv'
    print('cnn_trainable:',cnn_trainable)
    print('cnn dropout : ', cnn_dropout)
    print('dense dropout : ', dense_dropout)
    print('lstm dropout :', lstm_dropout)

    if mode == "both":
        frames = True
        differences = True
    elif mode == "only_frames":
        frames = True
        differences = False
    elif mode == "only_differences":
        frames = False
        differences = True

    if frames:

        frames_input = Input(shape=(seq_len, size, size, 3),name='frames_input')
        frames_cnn = MobileNetV2( input_shape = (size,size,3), alpha=0.35, weights='imagenet', include_top = False)
        frames_cnn = Model( inputs=[frames_cnn.layers[0].input],outputs=[frames_cnn.layers[-30].output] ) # taking only upto block 13
        
        for layer in frames_cnn.layers:
            layer.trainable = cnn_trainable

        frames_cnn = TimeDistributed( frames_cnn,name='frames_CNN' )( frames_input )
        frames_cnn = TimeDistributed( LeakyReLU(alpha=0.1), name='leaky_relu_1_' )( frames_cnn)
        frames_cnn = TimeDistributed( Dropout(cnn_dropout, seed=seed) ,name='dropout_1_' )(frames_cnn)

        if lstm_type == 'sepconv':
            frames_lstm = SepConvLSTM2D( filters = 64, kernel_size=(3, 3), padding='same', return_sequences=False, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, name='SepConvLSTM2D_1', kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(frames_cnn)
        # elif lstm_type == 'conv':    
        #     frames_lstm = ConvLSTM2D( filters = 64, kernel_size=(3, 3), padding='same', return_sequences=False, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, name='ConvLSTM2D_1', kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(frames_cnn)
        # elif lstm_type == 'asepconv':    
        #     frames_lstm = AttenSepConvLSTM2D( filters = 64, kernel_size=(3, 3), padding='same', return_sequences=False, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, name='AttenSepConvLSTM2D_1', kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(frames_cnn)
        # else:
        #     raise Exception("lstm type not recognized!")

        frames_lstm = BatchNormalization( axis = -1 )(frames_lstm)
        
    if differences:

        frames_diff_input = Input(shape=(seq_len - frame_diff_interval, size, size, 3),name='frames_diff_input')
        frames_diff_cnn = MobileNetV2( input_shape=(size,size,3), alpha=0.35, weights='imagenet', include_top = False)
        frames_diff_cnn = Model( inputs = [frames_diff_cnn.layers[0].input], outputs = [frames_diff_cnn.layers[-30].output] ) # taking only upto block 13
    
        for layer in frames_diff_cnn.layers:
            layer.trainable = cnn_trainable
    
        frames_diff_cnn = TimeDistributed( frames_diff_cnn,name='frames_diff_CNN' )(frames_diff_input)
        frames_diff_cnn = TimeDistributed( LeakyReLU(alpha=0.1), name='leaky_relu_2_' )(frames_diff_cnn)
        frames_diff_cnn = TimeDistributed( Dropout(cnn_dropout, seed=seed) ,name='dropout_2_' )(frames_diff_cnn)

        if lstm_type == 'sepconv':
            frames_diff_lstm = SepConvLSTM2D( filters = 64, kernel_size=(3, 3), padding='same', return_sequences=False, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, name='SepConvLSTM2D_2', kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(frames_diff_cnn)
        elif lstm_type == 'conv':    
            frames_diff_lstm = ConvLSTM2D( filters = 64, kernel_size=(3, 3), padding='same', return_sequences=False, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, name='ConvLSTM2D_2', kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(frames_diff_cnn)
        elif lstm_type == 'asepconv':    
            frames_diff_lstm = AttenSepConvLSTM2D( filters = 64, kernel_size=(3, 3), padding='same', return_sequences=False, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, name='AttenSepConvLSTM2D_2', kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(frames_diff_cnn)
        else:
            raise Exception("lstm type not recognized!")

        frames_diff_lstm = BatchNormalization( axis = -1 )(frames_diff_lstm)

    if frames:
        frames_lstm = MaxPooling2D((2,2))(frames_lstm)
        x1 = Flatten()(frames_lstm) 
        x1 = Dense(64)(x1)
        x1 = LeakyReLU(alpha=0.1)(x1)
        
    if differences:
        frames_diff_lstm = MaxPooling2D((2,2))(frames_diff_lstm)
        x2 = Flatten()(frames_diff_lstm)
        x2 = Dense(64)(x2)
        x2 = LeakyReLU(alpha=0.1)(x2)
    
    if mode == "both":
        x = Concatenate(axis=-1)([x1, x2])
    elif mode == "only_frames":
        x = x1
    elif mode == "only_differences":
        x = x2

    x = Dropout(dense_dropout, seed = seed)(x) 
    x = Dense(16)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(dense_dropout, seed = seed)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    if mode == "both":
        model = Model(inputs=[frames_input, frames_diff_input], outputs=predictions)
    elif mode == "only_frames":
        model = Model(inputs=frames_input, outputs=predictions)
    elif mode == "only_differences":
        model = Model(inputs=frames_diff_input, outputs=predictions)

    return model

def getProposedModelM(
        size=224
        , seq_len=32
        , frame_diff_interval = 1
        # , cnn_weight = 'imagenet'
        ,cnn_trainable = True
        # , lstm_type='sepconv'
        , weight_decay = 2e-5
        , mode = "both"
        , cnn_dropout = 0.25
        , lstm_dropout = 0.25
        , dense_dropout = 0.3
        , seed = 42
):
    """parameters:
    size = height/width of each frame,
    seq_len = number of frames in each sequence,
    cnn_weight= None or 'imagenet'
    mode = "only_frames" or "only_differences" or "both"
       returns:
    model
    """
    print('cnn_trainable:',cnn_trainable)
    print('cnn dropout : ', cnn_dropout)
    print('dense dropout : ', dense_dropout)
    print('lstm dropout :', lstm_dropout)

    if mode == "both":
        frames = True
        differences = True
    elif mode == "only_frames":
        frames = True
        differences = False
    elif mode == "only_differences":
        frames = False
        differences = True

    if frames:

        frames_input = Input(shape=(seq_len, size, size, 3),name='frames_input')
        frames_cnn = MobileNetV2( input_shape = (size,size,3), alpha=0.35, weights='imagenet', include_top = False)
        frames_cnn = Model( inputs=[frames_cnn.layers[0].input],outputs=[frames_cnn.layers[-30].output] ) # taking only upto block 13
        
        for layer in frames_cnn.layers:
            layer.trainable = cnn_trainable

        frames_cnn = TimeDistributed( frames_cnn,name='frames_CNN' )( frames_input )
        frames_cnn = TimeDistributed( LeakyReLU(alpha=0.1), name='leaky_relu_1_' )( frames_cnn)
        frames_cnn = TimeDistributed( Dropout(cnn_dropout, seed=seed) ,name='dropout_1_' )(frames_cnn)

        frames_lstm = SepConvLSTM2D( filters = 64, kernel_size=(3, 3), padding='same', return_sequences=False, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, name='SepConvLSTM2D_1', kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(frames_cnn)

        frames_lstm = BatchNormalization( axis = -1 )(frames_lstm)
        
    if differences:

        frames_diff_input = Input(shape=(seq_len - frame_diff_interval, size, size, 3),name='frames_diff_input')
        frames_diff_cnn = MobileNetV2( input_shape=(size,size,3), alpha=0.35, weights='imagenet', include_top = False)
        frames_diff_cnn = Model( inputs = [frames_diff_cnn.layers[0].input], outputs = [frames_diff_cnn.layers[-30].output] ) # taking only upto block 13
    
        for layer in frames_diff_cnn.layers:
            layer.trainable = cnn_trainable
    
        frames_diff_cnn = TimeDistributed( frames_diff_cnn,name='frames_diff_CNN' )(frames_diff_input)
        frames_diff_cnn = TimeDistributed( LeakyReLU(alpha=0.1), name='leaky_relu_2_' )(frames_diff_cnn)
        frames_diff_cnn = TimeDistributed( Dropout(cnn_dropout, seed=seed) ,name='dropout_2_' )(frames_diff_cnn)

        frames_diff_lstm = SepConvLSTM2D( filters = 64, kernel_size=(3, 3), padding='same', return_sequences=False, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, name='SepConvLSTM2D_2', kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(frames_diff_cnn)

        frames_diff_lstm = BatchNormalization( axis = -1 )(frames_diff_lstm)

    if frames:
        frames_lstm = MaxPooling2D((2,2))(frames_lstm)
        x1 = LeakyReLU(alpha=0.1)(frames_lstm)
        
    if differences:
        frames_diff_lstm = MaxPooling2D((2,2))(frames_diff_lstm)
        x2 = Activation("sigmoid")(frames_diff_lstm)
    
    if mode == "both":
        x = Multiply()([x1, x2])
    elif mode == "only_frames":
        x = x1
    elif mode == "only_differences":
        x = x2

    x = Flatten()(x)
    x = Dense(32)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(16)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(dense_dropout, seed = seed)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    if mode == "both":
        model = Model(inputs=[frames_input, frames_diff_input], outputs=predictions)
    elif mode == "only_frames":
        model = Model(inputs=frames_input, outputs=predictions)
    elif mode == "only_differences":
        model = Model(inputs=frames_diff_input, outputs=predictions)

    return model