from keras.layers import Convolution2D, Dense, Dropout, MaxPooling2D, Input, Flatten, merge, AveragePooling2D, \
    BatchNormalization, Activation, ZeroPadding2D
from keras.regularizers import l2
from keras.initializations import normal
from keras.engine.topology import Layer


class LRN2D(Layer):
    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        if n % 2 == 0:
            raise NotImplementedError("LRN2D only works with odd n. n provided: " + str(n))
        super(LRN2D, self).__init__(**kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def get_output(self, train):
        X = self.get_input(train)
        b, ch, r, c = K.shape(X)
        half_n = self.n // 2
        input_sqr = K.square(X)
        extra_channels = K.zeros((b, ch + 2 * half_n, r, c))
        input_sqr = K.concatenate([extra_channels[:, :half_n, :, :],
                                   input_sqr,
                                   extra_channels[:, half_n + ch:, :, :]],
                                  axis=1)
        scale = self.k
        for eye in range(self.n):
            scale += self.alpha * input_sqr[:, eye:eye + ch, :, :]
        scale **= self.beta
        return X / scale

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PoolHelper(Layer):

    def __init__(self, **kwargs):
        super(PoolHelper, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[:,1:,1:,:]

    def get_config(self):
        config = {}
        base_config = super(PoolHelper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AlexNet:
    def __init__(self):
        self.x = Input(shape=(227, 227, 3))
        y1 = Convolution2D(48, 11, 11, subsample=(4, 4), W_regularizer=l2(0.0005), border_mode='valid',
                           activation='relu', init=self.alexnet_norm)(self.x)
        y2 = Convolution2D(48, 11, 11, subsample=(4, 4), W_regularizer=l2(0.0005), border_mode='valid',
                           activation='relu', init=self.alexnet_norm)(self.x)
        y1 = LRN2D()(y1)
        y2 = LRN2D()(y2)
        y1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid')(y1)
        y2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid')(y2)
        y1 = Convolution2D(128, 5, 5, W_regularizer=l2(0.0005), border_mode='same', activation='relu',
                           init=self.alexnet_norm)(y1)
        # biases are 0.1
        y2 = Convolution2D(128, 5, 5, W_regularizer=l2(0.0005), border_mode='same', activation='relu',
                           init=self.alexnet_norm)(y2)
        # biases are 0.1
        y = merge(inputs=[y1, y2], mode='concat', concat_axis=3)
        y = LRN2D()(y)
        y = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid')(y)
        y1 = Convolution2D(192, 3, 3, W_regularizer=l2(0.0005), border_mode='same', activation='relu',
                           init=self.alexnet_norm)(y)
        y2 = Convolution2D(192, 3, 3, W_regularizer=l2(0.0005), border_mode='same', activation='relu',
                           init=self.alexnet_norm)(y)
        y1 = Convolution2D(192, 3, 3, W_regularizer=l2(0.0005), border_mode='same', activation='relu',
                           init=self.alexnet_norm)(y1)
        # biases are 0.1
        y2 = Convolution2D(192, 3, 3, W_regularizer=l2(0.0005), border_mode='same', activation='relu',
                           init=self.alexnet_norm)(y2)
        # biases are 0.1
        y1 = Convolution2D(128, 3, 3, W_regularizer=l2(0.0005), border_mode='same', activation='relu',
                           init=self.alexnet_norm)(y1)
        # biases are 0.1
        y2 = Convolution2D(128, 3, 3, W_regularizer=l2(0.0005), border_mode='same', activation='relu',
                           init=self.alexnet_norm)(y2)
        # biases are 0.1
        y = merge(inputs=[y1, y2], mode='concat', concat_axis=3)
        y = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid')(y)
        y = Flatten()(y)
        y = Dense(4096, activation='relu', W_regularizer=l2(0.0005),
                  init=lambda shape, name: normal(shape, scale=0.005, name=name))(y)  # biases are 0.1
        y = Dropout(0.5)(y)
        y = Dense(4096, activation='relu', W_regularizer=l2(0.0005),
                  init=lambda shape, name: normal(shape, scale=0.005, name=name))(y)  # biases are 0.1
        y = Dropout(0.5)(y)
        self.y = Dense(1000, activation='softmax', W_regularizer=l2(0.0005),
                       init=lambda shape, name: normal(shape, scale=0.01, name=name))(y)
        self.y_ = Input(shape=(1000,))

    @staticmethod
    def alexnet_norm(shape, name, dim_ordering):
        return normal(shape, scale=0.01, name=name, dim_ordering=dim_ordering)


class InceptionV3:
    def __init__(self):
        self.x = Input(shape=(224, 224, 3))
        self.y_ = Input(shape=(1000,))
        y = self.conv2d_bn(self.x, 32, 3, 3, subsample=(2, 2), border_mode='valid')
        y = self.conv2d_bn(y, 32, 3, 3, border_mode='valid')
        y = self.conv2d_bn(y, 64, 3, 3)
        y = MaxPooling2D((3, 3), strides=(2, 2))(y)
        y = self.conv2d_bn(y, 80, 1, 1, border_mode='valid')
        y = self.conv2d_bn(y, 192, 3, 3, border_mode='valid')
        y = MaxPooling2D((3, 3), strides=(2, 2))(y)
        y = self.inception135p(y, [[64], [64, 96, 96], [48, 64], [32]])
        y = self.inception135p(y, [[64], [64, 96, 96], [48, 64], [32]])
        y = self.inception135p(y, [[64], [64, 96, 96], [48, 64], [32]])
        y = self.inception33p(y, [[384], [64, 96, 96]])
        y = self.inception177p(y, [[192], [128, 128, 192], [128, 128, 128, 128, 128], [192]])
        y = self.inception177p(y, [[192], [160, 160, 192], [160, 160, 160, 160, 192], [192]])
        y = self.inception177p(y, [[192], [160, 160, 192], [160, 160, 160, 160, 192], [192]])
        y = self.inception177p(y, [[192], [192, 192, 192], [160, 192, 192, 192, 192], [192]])
        y = self.inception17p(y, [[192, 320], [192, 192, 192, 192]])
        y = self.inception133p(y, [[320], [384, 384, 384, ], [448, 384, 384, 384], [192]])
        y = self.inception133p(y, [[320], [384, 384, 384, ], [448, 384, 384, 384], [192]])
        # y = AveragePooling2D((8, 8), strides=(8, 8))(y)
        y = AveragePooling2D((5, 5), strides=(5, 5))(y)
        y = Flatten()(y)
        self.y = Dense(1000, activation='softmax')(y)

    @staticmethod
    def conv2d_bn(x, nb_filter, nb_row, nb_col, border_mode='same', subsample=(1, 1)):
        x = Convolution2D(nb_filter, nb_row, nb_col, subsample=subsample, activation='relu', border_mode=border_mode)(x)
        x = BatchNormalization(axis=3)(x)
        return x

    def inception135p(self, input_layer, params):
        p1, p2, p3, p4 = params
        branch1 = self.conv2d_bn(input_layer, p1[0], 1, 1)

        branch3 = self.conv2d_bn(input_layer, p2[0], 1, 1)
        branch3 = self.conv2d_bn(branch3, p2[1], 3, 3)
        branch3 = self.conv2d_bn(branch3, p2[2], 3, 3)

        branch5 = self.conv2d_bn(input_layer, p3[0], 1, 1)
        branch5 = self.conv2d_bn(branch5, p3[1], 5, 5)

        branchp = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input_layer)
        branchp = self.conv2d_bn(branchp, p4[0], 1, 1)
        return merge([branch1, branch3, branch5, branchp], mode='concat', concat_axis=3)

    def inception33p(self, input_layer, params):
        p1, p2 = params
        branch3a = self.conv2d_bn(input_layer, p1[0], 3, 3, subsample=(2, 2), border_mode='valid')

        branch3b = self.conv2d_bn(input_layer, p2[0], 1, 1)
        branch3b = self.conv2d_bn(branch3b, p2[1], 3, 3)
        branch3b = self.conv2d_bn(branch3b, p2[2], 3, 3, subsample=(2, 2), border_mode='valid')

        branchp = MaxPooling2D((3, 3), strides=(2, 2))(input_layer)

        return merge([branch3a, branch3b, branchp], mode='concat', concat_axis=3)

    def inception177p(self, input_layer, params):
        p1, p2, p3, p4 = params

        branch1 = self.conv2d_bn(input_layer, p1[0], 1, 1)

        branch2 = self.conv2d_bn(input_layer, p2[0], 1, 1)
        branch2 = self.conv2d_bn(branch2, p2[1], 1, 7)
        branch2 = self.conv2d_bn(branch2, p2[2], 7, 1)

        branch3 = self.conv2d_bn(input_layer, p3[0], 1, 1)
        branch3 = self.conv2d_bn(branch3, p3[1], 7, 1)
        branch3 = self.conv2d_bn(branch3, p3[2], 1, 7)
        branch3 = self.conv2d_bn(branch3, p3[3], 7, 1)
        branch3 = self.conv2d_bn(branch3, p3[4], 1, 7)

        branch4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input_layer)
        branch4 = self.conv2d_bn(branch4, p4[0], 1, 1)
        return merge([branch1, branch2, branch3, branch4], mode='concat', concat_axis=3)

    def inception17p(self, input_layer, params):
        p1, p2 = params

        branch1 = self.conv2d_bn(input_layer, p1[0], 1, 1)
        branch1 = self.conv2d_bn(branch1, p1[1], 3, 3, subsample=(2, 2), border_mode='valid')

        branch2 = self.conv2d_bn(input_layer, p2[0], 1, 1)
        branch2 = self.conv2d_bn(branch2, p2[1], 1, 7)
        branch2 = self.conv2d_bn(branch2, p2[2], 7, 7)
        branch2 = self.conv2d_bn(branch2, p2[3], 3, 3, subsample=(2, 2), border_mode='valid')

        branchp = AveragePooling2D((3, 3), strides=(2, 2))(input_layer)
        return merge([branch1, branch2, branchp], mode='concat', concat_axis=3)

    def inception133p(self, input_layer, params):
        p1, p2, p3, p4 = params

        branch1 = self.conv2d_bn(input_layer, p1[0], 1, 1)

        branch2 = self.conv2d_bn(input_layer, p2[0], 1, 1)
        branch2_1 = self.conv2d_bn(branch2, p2[1], 1, 3)
        branch2_2 = self.conv2d_bn(branch2, p2[2], 3, 1)
        branch2 = merge([branch2_1, branch2_2], mode='concat', concat_axis=3)

        branch3 = self.conv2d_bn(input_layer, p3[0], 1, 1)
        branch3 = self.conv2d_bn(branch3, p3[1], 3, 3)
        branch3_1 = self.conv2d_bn(branch3, p3[2], 1, 3)
        branch3_2 = self.conv2d_bn(branch3, p3[3], 3, 1)
        branch3 = merge([branch3_1, branch3_2], mode='concat', concat_axis=3)

        branchp = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input_layer)
        branchp = self.conv2d_bn(branchp, p4[0], 1, 1)
        return merge([branch1, branch2, branch3, branchp], mode='concat', concat_axis=3)


class ResNet50:
    def __init__(self):
        self.x = Input(shape=(224, 224, 3))
        self.y_ = Input(shape=(1000,))

        y = ZeroPadding2D((3, 3))(self.x)
        y = Convolution2D(64, 7, 7, subsample=(2, 2))(y)
        y = BatchNormalization(axis=3)(y)
        y = Activation('relu')(y)
        y = MaxPooling2D((3, 3), strides=(2, 2))(y)

        y = residual_module(y, 3, [64, 64, 256], shortcut=True, strides=(1, 1))
        y = residual_module(y, 3, [64, 64, 256])
        y = residual_module(y, 3, [64, 64, 256])

        y = residual_module(y, 3, [128, 128, 512], shortcut=True)
        y = residual_module(y, 3, [128, 128, 512])
        y = residual_module(y, 3, [128, 128, 512])
        y = residual_module(y, 3, [128, 128, 512])

        y = residual_module(y, 3, [256, 256, 1024], shortcut=True)
        y = residual_module(y, 3, [256, 256, 1024])
        y = residual_module(y, 3, [256, 256, 1024])
        y = residual_module(y, 3, [256, 256, 1024])
        y = residual_module(y, 3, [256, 256, 1024])
        y = residual_module(y, 3, [256, 256, 1024])

        y = residual_module(y, 3, [512, 512, 2048], shortcut=True)
        y = residual_module(y, 3, [512, 512, 2048])
        y = residual_module(y, 3, [512, 512, 2048])

        y = AveragePooling2D((7, 7))(y)

        y = Flatten()(y)
        self.y = Dense(1000, activation='softmax')(y)


class GoogLeNet:
    def __init__(self):
        self.x = Input(shape=(224, 224, 3))
        conv1_7x7_s2 = Convolution2D(64, 7, 7, subsample=(2, 2), border_mode='same', activation='relu',
                                     name='conv1/7x7_s2', W_regularizer=l2(0.0002))(self.x)
        conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)
        pool1_helper = PoolHelper()(conv1_zero_pad)
        pool1_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid', name='pool1/3x3_s2')(
            pool1_helper)
        pool1_norm1 = LRN2D(name='pool1/norm1')(pool1_3x3_s2)
        conv2_3x3_reduce = Convolution2D(64, 1, 1, border_mode='same', activation='relu', name='conv2/3x3_reduce',
                                         W_regularizer=l2(0.0002))(pool1_norm1)
        conv2_3x3 = Convolution2D(192, 3, 3, border_mode='same', activation='relu', name='conv2/3x3',
                                  W_regularizer=l2(0.0002))(conv2_3x3_reduce)
        conv2_norm2 = LRN2D(name='conv2/norm2')(conv2_3x3)
        conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_norm2)
        pool2_helper = PoolHelper()(conv2_zero_pad)
        pool2_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid', name='pool2/3x3_s2')(
            pool2_helper)

        inception_3a_1x1 = Convolution2D(64, 1, 1, border_mode='same', activation='relu', name='inception_3a/1x1',
                                         W_regularizer=l2(0.0002))(pool2_3x3_s2)
        inception_3a_3x3_reduce = Convolution2D(96, 1, 1, border_mode='same', activation='relu',
                                                name='inception_3a/3x3_reduce', W_regularizer=l2(0.0002))(pool2_3x3_s2)
        inception_3a_3x3 = Convolution2D(128, 3, 3, border_mode='same', activation='relu', name='inception_3a/3x3',
                                         W_regularizer=l2(0.0002))(inception_3a_3x3_reduce)
        inception_3a_5x5_reduce = Convolution2D(16, 1, 1, border_mode='same', activation='relu',
                                                name='inception_3a/5x5_reduce', W_regularizer=l2(0.0002))(pool2_3x3_s2)
        inception_3a_5x5 = Convolution2D(32, 5, 5, border_mode='same', activation='relu', name='inception_3a/5x5',
                                         W_regularizer=l2(0.0002))(inception_3a_5x5_reduce)
        inception_3a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same',
                                         name='inception_3a/pool')(pool2_3x3_s2)
        inception_3a_pool_proj = Convolution2D(32, 1, 1, border_mode='same', activation='relu',
                                               name='inception_3a/pool_proj', W_regularizer=l2(0.0002))(
            inception_3a_pool)
        inception_3a_output = merge([inception_3a_1x1, inception_3a_3x3, inception_3a_5x5, inception_3a_pool_proj],
                                    mode='concat', concat_axis=3, name='inception_3a/output')

        inception_3b_1x1 = Convolution2D(128, 1, 1, border_mode='same', activation='relu', name='inception_3b/1x1',
                                         W_regularizer=l2(0.0002))(inception_3a_output)
        inception_3b_3x3_reduce = Convolution2D(128, 1, 1, border_mode='same', activation='relu',
                                                name='inception_3b/3x3_reduce', W_regularizer=l2(0.0002))(
            inception_3a_output)
        inception_3b_3x3 = Convolution2D(192, 3, 3, border_mode='same', activation='relu', name='inception_3b/3x3',
                                         W_regularizer=l2(0.0002))(inception_3b_3x3_reduce)
        inception_3b_5x5_reduce = Convolution2D(32, 1, 1, border_mode='same', activation='relu',
                                                name='inception_3b/5x5_reduce', W_regularizer=l2(0.0002))(
            inception_3a_output)
        inception_3b_5x5 = Convolution2D(96, 5, 5, border_mode='same', activation='relu', name='inception_3b/5x5',
                                         W_regularizer=l2(0.0002))(inception_3b_5x5_reduce)
        inception_3b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same',
                                         name='inception_3b/pool')(inception_3a_output)
        inception_3b_pool_proj = Convolution2D(64, 1, 1, border_mode='same', activation='relu',
                                               name='inception_3b/pool_proj', W_regularizer=l2(0.0002))(
            inception_3b_pool)
        inception_3b_output = merge([inception_3b_1x1, inception_3b_3x3, inception_3b_5x5, inception_3b_pool_proj],
                                    mode='concat', concat_axis=3, name='inception_3b/output')

        inception_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_output)
        pool3_helper = PoolHelper()(inception_3b_output_zero_pad)
        pool3_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid', name='pool3/3x3_s2')(
            pool3_helper)

        inception_4a_1x1 = Convolution2D(192, 1, 1, border_mode='same', activation='relu', name='inception_4a/1x1',
                                         W_regularizer=l2(0.0002))(pool3_3x3_s2)
        inception_4a_3x3_reduce = Convolution2D(96, 1, 1, border_mode='same', activation='relu',
                                                name='inception_4a/3x3_reduce', W_regularizer=l2(0.0002))(pool3_3x3_s2)
        inception_4a_3x3 = Convolution2D(208, 3, 3, border_mode='same', activation='relu', name='inception_4a/3x3',
                                         W_regularizer=l2(0.0002))(inception_4a_3x3_reduce)
        inception_4a_5x5_reduce = Convolution2D(16, 1, 1, border_mode='same', activation='relu',
                                                name='inception_4a/5x5_reduce', W_regularizer=l2(0.0002))(pool3_3x3_s2)
        inception_4a_5x5 = Convolution2D(48, 5, 5, border_mode='same', activation='relu', name='inception_4a/5x5',
                                         W_regularizer=l2(0.0002))(inception_4a_5x5_reduce)
        inception_4a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same',
                                         name='inception_4a/pool')(pool3_3x3_s2)
        inception_4a_pool_proj = Convolution2D(64, 1, 1, border_mode='same', activation='relu',
                                               name='inception_4a/pool_proj', W_regularizer=l2(0.0002))(
            inception_4a_pool)
        inception_4a_output = merge([inception_4a_1x1, inception_4a_3x3, inception_4a_5x5, inception_4a_pool_proj],
                                    mode='concat', concat_axis=3, name='inception_4a/output')

        loss1_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss1/ave_pool')(inception_4a_output)
        loss1_conv = Convolution2D(128, 1, 1, border_mode='same', activation='relu', name='loss1/conv',
                                   W_regularizer=l2(0.0002))(loss1_ave_pool)
        loss1_flat = Flatten()(loss1_conv)
        loss1_fc = Dense(1024, activation='relu', name='loss1/fc', W_regularizer=l2(0.0002))(loss1_flat)
        loss1_drop_fc = Dropout(0.7)(loss1_fc)
        loss1_classifier = Dense(1000, name='loss1/classifier', W_regularizer=l2(0.0002))(loss1_drop_fc)
        self.y_aux = Activation('softmax')(loss1_classifier)

        inception_4b_1x1 = Convolution2D(160, 1, 1, border_mode='same', activation='relu', name='inception_4b/1x1',
                                         W_regularizer=l2(0.0002))(inception_4a_output)
        inception_4b_3x3_reduce = Convolution2D(112, 1, 1, border_mode='same', activation='relu',
                                                name='inception_4b/3x3_reduce', W_regularizer=l2(0.0002))(
            inception_4a_output)
        inception_4b_3x3 = Convolution2D(224, 3, 3, border_mode='same', activation='relu', name='inception_4b/3x3',
                                         W_regularizer=l2(0.0002))(inception_4b_3x3_reduce)
        inception_4b_5x5_reduce = Convolution2D(24, 1, 1, border_mode='same', activation='relu',
                                                name='inception_4b/5x5_reduce', W_regularizer=l2(0.0002))(
            inception_4a_output)
        inception_4b_5x5 = Convolution2D(64, 5, 5, border_mode='same', activation='relu', name='inception_4b/5x5',
                                         W_regularizer=l2(0.0002))(inception_4b_5x5_reduce)
        inception_4b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same',
                                         name='inception_4b/pool')(inception_4a_output)
        inception_4b_pool_proj = Convolution2D(64, 1, 1, border_mode='same', activation='relu',
                                               name='inception_4b/pool_proj', W_regularizer=l2(0.0002))(
            inception_4b_pool)
        inception_4b_output = merge([inception_4b_1x1, inception_4b_3x3, inception_4b_5x5, inception_4b_pool_proj],
                                    mode='concat', concat_axis=3, name='inception_4b_output')

        inception_4c_1x1 = Convolution2D(128, 1, 1, border_mode='same', activation='relu', name='inception_4c/1x1',
                                         W_regularizer=l2(0.0002))(inception_4b_output)
        inception_4c_3x3_reduce = Convolution2D(128, 1, 1, border_mode='same', activation='relu',
                                                name='inception_4c/3x3_reduce', W_regularizer=l2(0.0002))(
            inception_4b_output)
        inception_4c_3x3 = Convolution2D(256, 3, 3, border_mode='same', activation='relu', name='inception_4c/3x3',
                                         W_regularizer=l2(0.0002))(inception_4c_3x3_reduce)
        inception_4c_5x5_reduce = Convolution2D(24, 1, 1, border_mode='same', activation='relu',
                                                name='inception_4c/5x5_reduce', W_regularizer=l2(0.0002))(
            inception_4b_output)
        inception_4c_5x5 = Convolution2D(64, 5, 5, border_mode='same', activation='relu', name='inception_4c/5x5',
                                         W_regularizer=l2(0.0002))(inception_4c_5x5_reduce)
        inception_4c_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same',
                                         name='inception_4c/pool')(inception_4b_output)
        inception_4c_pool_proj = Convolution2D(64, 1, 1, border_mode='same', activation='relu',
                                               name='inception_4c/pool_proj', W_regularizer=l2(0.0002))(
            inception_4c_pool)
        inception_4c_output = merge([inception_4c_1x1, inception_4c_3x3, inception_4c_5x5, inception_4c_pool_proj],
                                    mode='concat', concat_axis=3, name='inception_4c/output')

        inception_4d_1x1 = Convolution2D(112, 1, 1, border_mode='same', activation='relu', name='inception_4d/1x1',
                                         W_regularizer=l2(0.0002))(inception_4c_output)
        inception_4d_3x3_reduce = Convolution2D(144, 1, 1, border_mode='same', activation='relu',
                                                name='inception_4d/3x3_reduce', W_regularizer=l2(0.0002))(
            inception_4c_output)
        inception_4d_3x3 = Convolution2D(288, 3, 3, border_mode='same', activation='relu', name='inception_4d/3x3',
                                         W_regularizer=l2(0.0002))(inception_4d_3x3_reduce)
        inception_4d_5x5_reduce = Convolution2D(32, 1, 1, border_mode='same', activation='relu',
                                                name='inception_4d/5x5_reduce', W_regularizer=l2(0.0002))(
            inception_4c_output)
        inception_4d_5x5 = Convolution2D(64, 5, 5, border_mode='same', activation='relu', name='inception_4d/5x5',
                                         W_regularizer=l2(0.0002))(inception_4d_5x5_reduce)
        inception_4d_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same',
                                         name='inception_4d/pool')(inception_4c_output)
        inception_4d_pool_proj = Convolution2D(64, 1, 1, border_mode='same', activation='relu',
                                               name='inception_4d/pool_proj', W_regularizer=l2(0.0002))(
            inception_4d_pool)
        inception_4d_output = merge([inception_4d_1x1, inception_4d_3x3, inception_4d_5x5, inception_4d_pool_proj],
                                    mode='concat', concat_axis=3, name='inception_4d/output')

        loss2_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss2/ave_pool')(inception_4d_output)
        loss2_conv = Convolution2D(128, 1, 1, border_mode='same', activation='relu', name='loss2/conv',
                                   W_regularizer=l2(0.0002))(loss2_ave_pool)
        loss2_flat = Flatten()(loss2_conv)
        loss2_fc = Dense(1024, activation='relu', name='loss2/fc', W_regularizer=l2(0.0002))(loss2_flat)
        loss2_drop_fc = Dropout(0.7)(loss2_fc)
        loss2_classifier = Dense(1000, name='loss2/classifier', W_regularizer=l2(0.0002))(loss2_drop_fc)
        self.y_aux2 = Activation('softmax')(loss2_classifier)

        inception_4e_1x1 = Convolution2D(256, 1, 1, border_mode='same', activation='relu', name='inception_4e/1x1',
                                         W_regularizer=l2(0.0002))(inception_4d_output)
        inception_4e_3x3_reduce = Convolution2D(160, 1, 1, border_mode='same', activation='relu',
                                                name='inception_4e/3x3_reduce', W_regularizer=l2(0.0002))(
            inception_4d_output)
        inception_4e_3x3 = Convolution2D(320, 3, 3, border_mode='same', activation='relu', name='inception_4e/3x3',
                                         W_regularizer=l2(0.0002))(inception_4e_3x3_reduce)
        inception_4e_5x5_reduce = Convolution2D(32, 1, 1, border_mode='same', activation='relu',
                                                name='inception_4e/5x5_reduce', W_regularizer=l2(0.0002))(
            inception_4d_output)
        inception_4e_5x5 = Convolution2D(128, 5, 5, border_mode='same', activation='relu', name='inception_4e/5x5',
                                         W_regularizer=l2(0.0002))(inception_4e_5x5_reduce)
        inception_4e_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same',
                                         name='inception_4e/pool')(inception_4d_output)
        inception_4e_pool_proj = Convolution2D(128, 1, 1, border_mode='same', activation='relu',
                                               name='inception_4e/pool_proj', W_regularizer=l2(0.0002))(
            inception_4e_pool)
        inception_4e_output = merge([inception_4e_1x1, inception_4e_3x3, inception_4e_5x5, inception_4e_pool_proj],
                                    mode='concat', concat_axis=3, name='inception_4e/output')

        inception_4e_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_output)
        pool4_helper = PoolHelper()(inception_4e_output_zero_pad)
        pool4_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid', name='pool4/3x3_s2')(
            pool4_helper)

        inception_5a_1x1 = Convolution2D(256, 1, 1, border_mode='same', activation='relu', name='inception_5a/1x1',
                                         W_regularizer=l2(0.0002))(pool4_3x3_s2)
        inception_5a_3x3_reduce = Convolution2D(160, 1, 1, border_mode='same', activation='relu',
                                                name='inception_5a/3x3_reduce', W_regularizer=l2(0.0002))(pool4_3x3_s2)
        inception_5a_3x3 = Convolution2D(320, 3, 3, border_mode='same', activation='relu', name='inception_5a/3x3',
                                         W_regularizer=l2(0.0002))(inception_5a_3x3_reduce)
        inception_5a_5x5_reduce = Convolution2D(32, 1, 1, border_mode='same', activation='relu',
                                                name='inception_5a/5x5_reduce', W_regularizer=l2(0.0002))(pool4_3x3_s2)
        inception_5a_5x5 = Convolution2D(128, 5, 5, border_mode='same', activation='relu', name='inception_5a/5x5',
                                         W_regularizer=l2(0.0002))(inception_5a_5x5_reduce)

        inception_5a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same',
                                         name='inception_5a/pool')(pool4_3x3_s2)
        inception_5a_pool_proj = Convolution2D(128, 1, 1, border_mode='same', activation='relu',
                                               name='inception_5a/pool_proj', W_regularizer=l2(0.0002))(
            inception_5a_pool)
        inception_5a_output = merge([inception_5a_1x1, inception_5a_3x3, inception_5a_5x5, inception_5a_pool_proj],
                                    mode='concat', concat_axis=3, name='inception_5a/output')
        inception_5b_1x1 = Convolution2D(384, 1, 1, border_mode='same', activation='relu', name='inception_5b/1x1',
                                         W_regularizer=l2(0.0002))(inception_5a_output)
        inception_5b_3x3_reduce = Convolution2D(192, 1, 1, border_mode='same', activation='relu',
                                                name='inception_5b/3x3_reduce', W_regularizer=l2(0.0002))(
            inception_5a_output)
        inception_5b_3x3 = Convolution2D(384, 3, 3, border_mode='same', activation='relu', name='inception_5b/3x3',
                                         W_regularizer=l2(0.0002))(inception_5b_3x3_reduce)
        inception_5b_5x5_reduce = Convolution2D(48, 1, 1, border_mode='same', activation='relu',
                                                name='inception_5b/5x5_reduce', W_regularizer=l2(0.0002))(
            inception_5a_output)
        inception_5b_5x5 = Convolution2D(128, 5, 5, border_mode='same', activation='relu', name='inception_5b/5x5',
                                         W_regularizer=l2(0.0002))(inception_5b_5x5_reduce)
        inception_5b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same',
                                         name='inception_5b/pool')(inception_5a_output)
        inception_5b_pool_proj = Convolution2D(128, 1, 1, border_mode='same', activation='relu',
                                               name='inception_5b/pool_proj', W_regularizer=l2(0.0002))(
            inception_5b_pool)
        inception_5b_output = merge([inception_5b_1x1, inception_5b_3x3, inception_5b_5x5, inception_5b_pool_proj],
                                    mode='concat', concat_axis=3, name='inception_5b/output')

        pool5_7x7_s1 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), name='pool5/7x7_s2')(inception_5b_output)
        loss3_flat = Flatten()(pool5_7x7_s1)
        pool5_drop_7x7_s1 = Dropout(0.4)(loss3_flat)
        loss3_classifier = Dense(1000, name='loss3/classifier', W_regularizer=l2(0.0002))(pool5_drop_7x7_s1)
        self.y = Activation('softmax', name='prob')(loss3_classifier)
        self.y_ = Input(shape=(1000,))


def residual_module(input_layer, kernel_size, filters, shortcut=False, strides=(2, 2)):
    filter1, filter2, filter3 = filters
    if shortcut:
        out = Convolution2D(filter1, 1, 1, subsample=strides)(input_layer)
    else:
        out = Convolution2D(filter1, 1, 1)(input_layer)
    out = BatchNormalization(axis=3)(out)
    out = Activation('relu')(out)

    out = Convolution2D(filter2, kernel_size, kernel_size, border_mode='same')(out)
    out = BatchNormalization(axis=3)(out)
    out = Activation('relu')(out)

    out = Convolution2D(filter3, 1, 1)(out)
    out = BatchNormalization(axis=3)(out)
    if shortcut:
        shortcut = Convolution2D(filter3, 1, 1, subsample=strides)(input_layer)
        shortcut = BatchNormalization(axis=3)(shortcut)
        out = merge([out, shortcut], mode='sum')
    else:
        out = merge([out, input_layer], mode='sum')
    out = Activation('relu')(out)
    return out


if __name__ == "__main__":
    net = AlexNet()
    from keras.models import Model
    model = Model(net.x, net.y)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()
