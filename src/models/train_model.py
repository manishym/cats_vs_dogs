from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras import backend as K
from keras.applications import VGG16
from keras import optimizers
import numpy as np


img_width, img_height = 150, 150

train_data_dir = "data/interim/train"
validation_data_dir = "data/interim/validation"
top_model_weights_path = "models/bottleneck_fc_model.h5"

nb_train_samples = 4700
nb_validation_samples = 222

epochs=50
batch_size = 16

if K.image_data_format() == "channels_first":
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


def create_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def train_and_test(train_data_dir=train_data_dir, validation_data_dir=validation_data_dir):
    model = create_model()
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    train_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="binary")

    validation_generator = train_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="binary")

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    model.save_weights("models/first_try.h5")


def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)
    model = VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)

    np.save(open(
        "models/bottleneck_features_train.npy", "wb"),
        bottleneck_features_train)
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open(
        "models/bottleneck_features_validation.npy", "wb"),
        bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open("models/bottleneck_features_train.npy", "rb"))
    train_labels = np.array(
        [0] * (2208//2) + [1] * (2208 // 2))
    validation_data = np.load(open("models/bottleneck_features_validation.npy", "rb"))
    validation_labels = np.array(
        [0] * (208 // 2) + [1] * (208 // 2))
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(validation_data, validation_labels), verbose=1)
    model.save_weights(top_model_weights_path)


def final_train_and_test():
    input_tensor = Input(shape=(150,150,3))
    vgg_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
    print("VGG loaded")
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation="sigmoid"))
    top_model.load_weights(top_model_weights_path)
    # model.add(top_model)
    model = Model(input=vgg_model.input, output=top_model(vgg_model.output))
    for layer in model.layers[:25]:
        layer.trainable = False
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
        metrics=["accuracy"])
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    # fine-tune the model
    model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        epochs=epochs,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,
        verbose=1)



def main():
    # train_and_test()
    # save_bottleneck_features()
    # train_top_model()
    final_train_and_test()
    # model = VGG16(weights='imagenet', include_top=False)
    # print(model.output_shape[:1])


if __name__ == '__main__':
    main()
