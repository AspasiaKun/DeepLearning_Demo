import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

def set_image_dir():
    origin_dataset_dir = '/home/aspasia/桌面/DeepLearningDemo/dogvscat/train'

    base_dir = '/home/aspasia/桌面/DeepLearningDemo/dogvscat/cats_and_dogs_small'
    os.mkdir(base_dir)

    train_dir = os.path.join(base_dir,'train')
    os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir,'validation')
    os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir,'test')
    os.mkdir(test_dir)

    train_cats_dir = os.path.join(train_dir,'cats')
    os.mkdir(train_cats_dir)
    train_dogs_dir = os.path.join(train_dir,'dogs')
    os.mkdir(train_dogs_dir)

    validation_cats_dir = os.path.join(validation_dir,'cats')
    os.mkdir(validation_cats_dir)
    validation_dogs_dir = os.path.join(validation_dir,'dogs')
    os.mkdir(validation_dogs_dir)

    test_cats_dir = os.path.join(test_dir,'cats')
    os.mkdir(test_cats_dir)
    test_dogs_dir = os.path.join(test_dir,'dogs')
    os.mkdir(test_dogs_dir)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(origin_dataset_dir,fname)
        dst = os.path.join(train_cats_dir,fname)
        shutil.copyfile(src,dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
    for fname in fnames:
        src = os.path.join(origin_dataset_dir,fname)
        dst = os.path.join(validation_cats_dir,fname)
        shutil.copyfile(src,dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
    for fname in fnames:
        src = os.path.join(origin_dataset_dir,fname)
        dst = os.path.join(test_cats_dir,fname)
        shutil.copyfile(src,dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(origin_dataset_dir,fname)
        dst = os.path.join(train_dogs_dir,fname)
        shutil.copyfile(src,dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000,1500)]
    for fname in fnames:
        src = os.path.join(origin_dataset_dir,fname)
        dst = os.path.join(validation_dogs_dir,fname)
        shutil.copyfile(src,dst)  

    fnames = ['dog.{}.jpg'.format(i) for i in range(1500,2000)]
    for fname in fnames:
        src = os.path.join(origin_dataset_dir,fname)
        dst = os.path.join(test_dogs_dir,fname)
        shutil.copyfile(src,dst)

def main():
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation='relu'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation='relu'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        "/home/aspasia/桌面/DeepLearningDemo/dogvscat/cats_and_dogs_small/train",
        target_size=(150,150),
        batch_size=20,
        class_mode='binary')
    
    validation_generator = test_datagen.flow_from_directory(
        '/home/aspasia/桌面/DeepLearningDemo/dogvscat/cats_and_dogs_small/validation',
        target_size=(150,150),
        batch_size=20,
        class_mode='binary')
    
    history = model.fit(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)
    
    model.save('cats_and_dogs_small_1.h5')

if __name__ == "__main__":
    main()






