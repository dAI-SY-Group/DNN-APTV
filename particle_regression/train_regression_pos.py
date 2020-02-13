import numpy as np
import pandas as pd
from keras import Model
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Input, Dense, concatenate
from keras.optimizers import Adam, RMSprop, Nadam
from keras.utils import Progbar
from keras_preprocessing.image import ImageDataGenerator
#from efficientnet import EfficientNetB0
from data_loader_regression_pos_z import multi_input_data_gen, get_git_revision_hash, get_class_form_data, \
    LearningRateExponentialDecay

# Training parameters
backbone = 'inceptionv3'  # densenet121,densenet169,densenet201,inceptionv3
#backbone = 'efficientnetb2'  # densenet121,densenet169,densenet201,inceptionv3,resnet50
#backbone = 'densenet201'  # densenet121,densenet169,densenet201,inceptionv3,resnet50
optimizer = 'adam'  # adam, rmsprop
loss = 'mae'
batch_size = 64
learning_rate = 0.01
data_augmentation = True
epochs = 200
image_HWC = (221, 221, 3)

if backbone == 'efficientnetb2':
    learning_rate = 0.016 * (batch_size / 256.)

model_name = 'regression_kali5_0-60_train_lr0001_model_%s_train_bs%d_%s_lr%s_%s_pos' % (backbone, batch_size, learning_rate, optimizer, loss)
model_save_path = '%s.hdf5' % model_name
log_save_path = '%s-train-%s.log' % (model_name, get_git_revision_hash())
if backbone == 'densenet121':
    BACKBONE = DenseNet121
elif backbone == 'densenet169':
    BACKBONE = DenseNet169
elif backbone == 'densenet201':
    BACKBONE = DenseNet201
elif backbone == 'inceptionv3':
    BACKBONE = InceptionV3
elif backbone == 'resnet50':
    BACKBONE = ResNet50
elif backbone == 'efficientnetb2':
    # noinspection PyUnresolvedReferences
    from efficientnet import EfficientNetB2
    BACKBONE = EfficientNetB2

if optimizer == 'adam':
    OPTIMIZER = Adam
elif optimizer == 'nadam':
    OPTIMIZER = Nadam
elif optimizer == 'rmsprop':
    OPTIMIZER = RMSprop

noAugGen = ImageDataGenerator(rescale=1. / 255,samplewise_center=True,samplewise_std_normalization=True,)
# Load data.
if data_augmentation:
    genImg = ImageDataGenerator(
        rescale=1. / 255,
        samplewise_center=True,
        samplewise_std_normalization=True,
        # randomly rotate images in the range (deg 0 to 180)
        # rotation_range=0.1,
        # randomly shift images horizontally
        width_shift_range=0.01,
        # randomly shift images vertically
        height_shift_range=0.01,
        # randomly flip images
        #horizontal_flip=True,
        # vertical_flip=True,
        brightness_range=[0.9, 1.1]
    )
else:
    genImg = ImageDataGenerator(rescale=1. / 255)
# genImg.flow_from_dataframe()
image_dir_train = 'pngkali5train'
image_dir_val = 'pngkali5val'
image_dir_test = 'pngkali5test'

train_gen = multi_input_data_gen(genImg, 'csv/kali5_train_regression.csv', image_dir_train,filter_border=100,color_mode='rgb',filter_range=1)
val_gen = multi_input_data_gen(noAugGen, 'csv/kali5_val_regression.csv', image_dir_val,filter_border=100,color_mode='rgb',filter_range=1)
test_gen_all = multi_input_data_gen(noAugGen, 'csv/kali5_test_regression.csv', image_dir_test, shuffle=False,filter_border=100,color_mode='rgb')


# Build model.
input = Input(shape=image_HWC)
input_pos_x = Input(shape=(1,))
input_pos_y = Input(shape=(1,))
base_model = BACKBONE(input_shape=image_HWC, include_top=False, weights=None, pooling='max')
x = base_model(input)
x = concatenate([x, input_pos_x, input_pos_y])
x = Dense(1024, activation='relu')(x)
x = Dense(1)(x)
model = Model([input, input_pos_x, input_pos_y], x)
model.compile(loss=loss, optimizer=OPTIMIZER(lr=learning_rate, decay=0.1))
model.summary()

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=model_save_path, monitor='val_loss', verbose=1, save_best_only=True)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
train_logger = CSVLogger(log_save_path)
# 'learning_rate_decay_factor': 0.94,
# 'num_epochs_per_decay': 2.4,
num_epochs_per_decay = 2.4
step_decay = len(train_gen) * num_epochs_per_decay
lr_exp = LearningRateExponentialDecay(0.94, step_decay)
callbacks = [checkpoint, lr_exp, train_logger]

try:
    model.load_weights(model_save_path)
    print('Loading pretrain_weights!')
except Exception as e:
    print(e)
    pass

print('Using real-time data augmentation.')
model.fit_generator(train_gen,
                    validation_data=val_gen,
                    validation_steps=len(val_gen),
                    epochs=epochs, workers=1,
                    steps_per_epoch=len(train_gen) / 8,
                    callbacks=callbacks)



def predict_on_model(gen_ds):
    y_test = []
    y_pred = []
    in_data_x=[]
    in_data_y=[]
    # use predict on batch instead of predict_generator https://github.com/keras-team/keras/issues/5048
    progbar = Progbar(target=len(gen_ds))
    for i in range(len(gen_ds)):
        indata, y_value = next(gen_ds)
        pred = model.predict_on_batch(indata)
        for posx in indata[1]:
            px=np.asarray([posx])
            in_data_x.append(px)
        for posy in indata[2]:
            qy=np.asarray([posy])
            in_data_y.append(qy)
        for p in pred:
            y_pred.append(p)
        for y in y_value:
            y_test.append(y)
        progbar.update(i + 1)
    return np.asarray(y_test), np.asarray(y_pred),  np.asarray(in_data_x) , np.asarray(in_data_y)

#model = keras.models.load_model(model_save_path)
model.load_weights(model_save_path)

y_test, y_pred ,in_data_x,in_data_y  = predict_on_model(test_gen_all)
print('TestData all mae: {0}.'.format(np.mean(np.abs(np.squeeze(y_pred) - y_test))))
data = pd.DataFrame({'posx':np.squeeze(in_data_x),'posy':np.squeeze(in_data_y),'y_ground_truth': np.squeeze(y_test), 'y_pred': np.squeeze(y_pred)})
data.to_csv('ygt_ypred_test_0-60_rmsprop_lr0001_efficientnb2_bs256_kalimix_with_position.csv')
