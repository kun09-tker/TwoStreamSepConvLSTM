from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Flatten, Dropout, ConvLSTM2D, BatchNormalization, Activation
from tensorflow.keras.layers import TimeDistributed, Multiply, Add
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import LeakyReLU, Conv3D, Multiply, MaxPooling3D, MaxPooling2D, Concatenate, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from .sep_conv_rnn import SepConvLSTM2D

def getProposedModelC(
        size=22
        , seq_len=32 
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
    size = height/width of each k part,
    seq_len = number of group of part heatmaps in each sequence,
    cnn_weight= None or 'imagenet'
    mode = "only_limb" or "only_keypoints" or "both"
       returns:
    model
    """
    # print('cnn_trainable:',cnn_trainable)
    # print('cnn dropout : ', cnn_dropout)
    # print('dense dropout : ', dense_dropout)
    # print('lstm dropout :', lstm_dropout)

    if mode == "both":
        limbs = True
        keypoints = True
    elif mode == "only_limbs":
        limbs = True
        keypoints = False
    elif mode == "only_keypoints":
        limbs = False
        keypoints = True

    if limbs:
        limbs_input = Input(shape=(seq_len, size, size, 3),name='limbs_input')
        limbs_cnn = MobileNetV2( input_shape = (size, size, 3), alpha=0.35, weights=None, include_top = False)
        limbs_cnn = Model( inputs=[limbs_cnn.layers[0].input],outputs=[limbs_cnn.layers[-10].output] ) # taking only upto block 13

        for layer in limbs_cnn.layers:
            layer.trainable = cnn_trainable

        limbs_cnn = TimeDistributed( limbs_cnn,name='limbs_CNN' )( limbs_input )
        limbs_cnn = TimeDistributed( LeakyReLU(alpha=0.1), name='leaky_relu_1_' )( limbs_cnn)
        limbs_cnn = TimeDistributed( Dropout(cnn_dropout, seed=seed) ,name='dropout_1_' )(limbs_cnn)

        limbs_lstm = SepConvLSTM2D( filters = 64, kernel_size=(3, 3), padding='same', return_sequences=False, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, name='SepConvLSTM2D_1', kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(limbs_cnn)
        limbs_lstm = BatchNormalization( axis = -1 )(limbs_lstm)

    if keypoints:
        keypoints_input = Input(shape=(seq_len, size, size, 3),name='keypoints_input')
        keypoints_cnn = MobileNetV2( input_shape=(size, size, 3), alpha=0.35, weights=None, include_top = False)
        keypoints_cnn = Model( inputs = [keypoints_cnn.layers[0].input], outputs = [keypoints_cnn.layers[-10].output])
    
        for layer in keypoints_cnn.layers:
            layer.trainable = cnn_trainable

        keypoints_cnn = TimeDistributed( keypoints_cnn,name='keypoints_CNN' )(keypoints_input)
        keypoints_cnn = TimeDistributed( LeakyReLU(alpha=0.1), name='leaky_relu_2_' )(keypoints_cnn)
        keypoints_cnn = TimeDistributed( Dropout(cnn_dropout, seed=seed) ,name='dropout_2_' )(keypoints_cnn)

        keypoints_lstm = SepConvLSTM2D( filters = 64, kernel_size=(3, 3), padding='same', return_sequences=False, dropout=lstm_dropout, recurrent_dropout=lstm_dropout, name='SepConvLSTM2D_2', kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(keypoints_cnn)
        keypoints_lstm = BatchNormalization( axis = -1 )(keypoints_lstm)

    if limbs:
        limbs_lstm = MaxPooling2D((2,2))(limbs_lstm)
        x1 = Flatten()(limbs_lstm) 
        x1 = Dense(64)(x1)
        x1 = LeakyReLU(alpha=0.1)(x1)
        
    if keypoints:
        keypoints_lstm = MaxPooling2D((2,2))(keypoints_lstm)
        x2 = Flatten()(keypoints_lstm)
        x2 = Dense(64)(x2)
        x2 = LeakyReLU(alpha=0.1)(x2)

    if mode == "both":
        x = Concatenate(axis=-1)([x1, x2])
    elif mode == "only_limbs":
        x = x1
    elif mode == "only_keypoints":
        x = x2

    x = Dropout(dense_dropout, seed = seed)(x) 
    x = Dense(16)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(dense_dropout, seed = seed)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    if mode == "both":
        model = Model(inputs=[limbs_input, keypoints_input], outputs=predictions)
    elif mode == "only_limbs":
        model = Model(inputs=limbs_input, outputs=predictions)
    elif mode == "only_keypoints":
        model = Model(inputs=keypoints_input, outputs=predictions)

    return model
