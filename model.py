import numpy as np

import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)
# tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Input,Conv2D,Flatten,Dense,Lambda,Reshape,Conv2DTranspose,MaxPooling2D,UpSampling2D
from tensorflow.keras.layers import concatenate,add
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MSE

class AE(tf.keras.Model):

    def __init__(self,img_shape=(256,256,3), latent_dim=256):
        super(AE, self).__init__(name='ae')
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.encoder, self.ae  = self.build_model()
    
    def build_model(self):
        ''' Encoder '''
        input_img = Input(shape=self.img_shape, name='encoder_input')

        l1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(input_img)
        l2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l1)
        l3 = MaxPooling2D(padding='same')(l2)
        l4 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l3)
        l5 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l4)
        l6 = MaxPooling2D(padding='same')(l5)
        l7 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l6)

        # 潜在空間部分
        shape = K.int_shape(l7)
        x = Flatten()(l7)
        # x = Dense(256, activation='relu')(x)
        z = Dense(self.latent_dim)(x)

        ''' Decoder '''
        x = Dense(shape[1]*shape[2]*shape[3], activation='relu')(z)
        x = Reshape((shape[1], shape[2], shape[3]))(x) # 最後のFlatten層の直前の特徴マップと同じ形状の特徴マップに変換

        l8  = UpSampling2D()(x)
        l9  = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l8)
        l10 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l9)
        l11 = concatenate([l10, l5])
        l12 = UpSampling2D()(l11)
        l13 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l12)
        l14 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l13)
        l15 = concatenate([l14, l2])    
        out = Conv2D(3, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l15)

        AutoEncoder = Model(inputs=input_img, outputs=out)
        encoder = Model(input_img, z)

        return encoder,AutoEncoder


    def get_encoder(self):
        return self.encoder


    def get_ae(self):
        self.ae.compile(optimizer='adam', loss="mse")#, experimental_run_tf_function=False)
        return self.ae
        
        
class VAE(tf.keras.Model):

    def __init__(self,img_shape=(256,256,3), latent_dim=256, beta=1):
        super(VAE, self).__init__(name='vae')
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.beta = beta
        self.z_m = None
        self.z_s = None

        self.encoder, self.decoder, self.vae  = self.build_model()


    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=1.)

        return z_mean + self.beta * K.exp(z_log_var) * epsilon

    
    def build_model(self):
        # Encoder
        input_img = Input(shape=self.img_shape)
        
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(input_img)
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(x)
        x = MaxPooling2D(padding='same')(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(x)
        x = MaxPooling2D(padding='same')(x)
        x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(x)
        
        shape_before_flattening = K.int_shape(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)

        z_mean = Dense(self.latent_dim)(x) # outputの次元数：潜在変数の数
        z_log_var = Dense(self.latent_dim)(x)

        self.z_m = z_mean  # for Loss
        self.z_s = z_log_var # for Loss

        z = Lambda(self.sampling)([z_mean, z_log_var])
        
        # Decoder
        decoder_input = Input(K.int_shape(z)[1:])
        x = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
        x = Reshape(shape_before_flattening[1:])(x)
        
        x  = UpSampling2D()(x)
        x  = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(x)
        x = UpSampling2D()(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(x)
        out = Conv2D(3, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(x)

        
        encoder = Model(input_img, z_mean)
        decoder = Model(decoder_input, out)
        
        z_decoded = decoder(z)
        vae = Model(input_img, z_decoded)

        # y = CustomVariationalLayer()([input_img, z_decoded, z_mean, z_log_var])
        # vae = Model(input_img, y)

        return encoder,decoder,vae


    def binary_crossentropy(self, y_true, y_pred):
        return K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)


    def vae_loss(self, x, x_decoded_mean):
        z_mean = self.z_m
        z_log_var = self.z_s

        latent_loss =  - 5e-4 * K.mean(K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))
        reconst_loss = K.mean(self.binary_crossentropy(x, x_decoded_mean),axis=-1)

        # z_sigma = self.z_s
        # latent_loss =  - 0.5 * K.mean(K.sum(1 + K.log(K.square(z_sigma)) - K.square(z_mean) - K.square(z_sigma), axis=-1))
        
        return latent_loss + reconst_loss

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_vae(self):
        self.vae.compile(optimizer='adam', loss=self.vae_loss)#, experimental_run_tf_function=False)
        return self.vae
        


class VAE2(tf.keras.Model):

    def __init__(self,img_shape, latent_dim, beta):
        super(VAE2, self).__init__(name='vae2')
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.beta = beta
        self.z_m = None
        self.z_s = None

        self.encoder, self.decoder, self.vae  = self.build_model()


    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=1.)

        return z_mean + self.beta * K.exp(z_log_var) * epsilon

    
    def build_model(self):
        # Encoder
        input_img = Input(shape=self.img_shape)
        x = Conv2D(32, 3, padding='same', activation='relu')(input_img)
        x = Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
        x = Conv2D(64, 3, padding='same', activation='relu')(x)
        x = Conv2D(64, 3, padding='same', activation='relu')(x)
        shape_before_flattening = K.int_shape(x)
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)

        z_mean = Dense(self.latent_dim)(x) # outputの次元数：潜在変数の数
        z_log_var = Dense(self.latent_dim)(x)

        self.z_m = z_mean  # for Loss
        self.z_s = z_log_var # for Loss

        z = Lambda(self.sampling)([z_mean, z_log_var])
        
        encoder = Model(input_img, z_mean)

        # Decoder
        decoder_input = Input(K.int_shape(z)[1:])
        x = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
        x = Reshape(shape_before_flattening[1:])(x)
        x = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
        x = Conv2D(3, 3, padding='same', activation='sigmoid')(x)

        decoder = Model(decoder_input, x)
        z_decoded = decoder(z)
    
        vae = Model(input_img, z_decoded)

        # y = CustomVariationalLayer()([input_img, z_decoded, z_mean, z_log_var])
        # vae = Model(input_img, y)

        return encoder,decoder,vae


    def binary_crossentropy(self, y_true, y_pred):
        return K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)


    def vae_loss(self, x, x_decoded_mean):
        z_mean = self.z_m
        z_log_var = self.z_s

        latent_loss =  - 5e-4 * K.mean(K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))
        reconst_loss = K.mean(self.binary_crossentropy(x, x_decoded_mean),axis=-1)

        # z_sigma = self.z_s
        # latent_loss =  - 0.5 * K.mean(K.sum(1 + K.log(K.square(z_sigma)) - K.square(z_mean) - K.square(z_sigma), axis=-1))
        
        return latent_loss + reconst_loss

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_vae(self):
        self.vae.compile(optimizer='adam', loss=self.vae_loss, experimental_run_tf_function=False)
        return self.vae
        



"""
    def get_AE(Z_dim=256, INPUT_SHAPE=(256, 256, 3)):

    ''' Encoder '''
    input_img = Input(shape=INPUT_SHAPE, name='encoder_input')

    l1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(input_img)
    l2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l1)
    l3 = MaxPooling2D(padding='same')(l2)
    l4 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l3)
    l5 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l4)
    l6 = MaxPooling2D(padding='same')(l5)
    l7 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l6)
    
    # 潜在空間部分
    shape = K.int_shape(l7)
    x = Flatten()(l7)
    # x = Dense(256, activation='relu')(x)
    z = Dense(Z_dim)(x)
    
    ''' Decoder '''
    x = Dense(shape[1]*shape[2]*shape[3], activation='relu')(z)
    x = Reshape((shape[1], shape[2], shape[3]))(x) # 最後のFlatten層の直前の特徴マップと同じ形状の特徴マップに変換
    
    l8  = UpSampling2D()(x)
    l9  = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l8)
    l10 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l9)
    l11 = concatenate([l10, l5])
    l12 = UpSampling2D()(l11)
    l13 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l12)
    l14 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l13)
    l15 = concatenate([l14, l2])    
    out = Conv2D(3, (3, 3), padding='same', kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10))(l15)

    AutoEncoder = Model(inputs=input_img, outputs=out)
    encoder = Model(input_img, z)
    
    return AutoEncoder, encoder
"""
