'''
Αλγόριθμος για Transfer Learning CNN Image Classification
New Plant Disease Dataset (Augmented)

'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import time

start_time = time.time()

#Ανακάλυψη των φυσικών συσκευών (physical devices) για training του NN
#Χρήση του GPU για training
physicalDevices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physicalDevices[0], True)

BATCH_SIZE  = 32   #τόσες είκόνες θα φορτόνωνται στο μοντέλο ανα φορά
IMG_SIZE  = (150, 150) 

#Ανάληψη των δεδομένων από το Dataset
#Data για training
trainDS = tf.keras.preprocessing.image_dataset_from_directory(
    'New Plant Diseases Dataset(Augmented)/train',
    seed=42,
    #validation_split=0.2,
    #subset="training",
    image_size= IMG_SIZE ,
    batch_size=BATCH_SIZE
)

#Split των δεδομώνεν σε 80% train και 20% validation
tr_batches = tf.data.experimental.cardinality(trainDS)
validateDS = trainDS.take(tr_batches // 5)
trainDS = trainDS.skip(tr_batches // 5)

#Data για validation
validateDS = tf.keras.preprocessing.image_dataset_from_directory(
    'New Plant Diseases Dataset(Augmented)/valid',
    seed=42,
    #validation_split=0.2,
    #subset="validation",
    image_size= IMG_SIZE,
    batch_size=BATCH_SIZE
)

#Split των δεδομώνεν σε 80% validation και 20% testing
val_batches = tf.data.experimental.cardinality(validateDS)
testDS = validateDS.take(val_batches // 5)
validateDS = validateDS.skip(val_batches // 5)

#Data για test
#testDS = tf.keras.preprocessing.image_dataset_from_directory(
  # 'test',
    #seed=42,
    #validation_split=0.2,
    #subset="validation",
    #image_size= IMG_SIZE,
    #batch_size=BATCH_SIZE
#)


#Ορισμός των κλάσεων ανα Dataset
#classes = trainDS.class_names
nClasses = 38

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

#Δημιουργία προεκπαιδευόμενων μοντέλων convnets

def resnet50(IMG_SHAPE):
    preprocess_input = tf.keras.applications.resnet.preprocess_input  #προεπεξεργασία των δεδομένων
#Βασικό μποντέλο
    base_model = tf.keras.applications.resnet50.ResNet50(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
    return base_model, preprocess_input

def resnet50v2(IMG_SHAPE):
    preprocess_input = tf.keras.applications.resnet_v2.preprocess_input  #προεπεξεργασία των δεδομένων
    base_model = tf.keras.applications.resnet_v2.ResNet50V2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
    return base_model, preprocess_input

def inceptionResnetv2(IMG_SHAPE):
    preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input  #προεπεξεργασία των δεδομένων
    base_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
    return base_model, preprocess_input

def mobilenetv2(IMG_SHAPE):
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input  #προεπεξεργασία των δεδομένων
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
    return base_model, preprocess_input

def inceptionv3(IMG_SHAPE):
    preprocess_input = tf.keras.applications.inception_v3.preprocess_input  #προεπεξεργασία των δεδομένων
    base_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
    return base_model, preprocess_input

def xception(IMG_SHAPE):
    preprocess_input = tf.keras.applications.xception.preprocess_input  #προεπεξεργασία των δεδομένων
    base_model = tf.keras.applications.xception.Xception(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
    return base_model, preprocess_input

def vgg16(IMG_SHAPE):
    preprocess_input = tf.keras.applications.vgg16.preprocess_input  #προεπεξεργασία των δεδομένων
    base_model = tf.keras.applications.vgg16.VGG16(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
    return base_model, preprocess_input

def vgg19(IMG_SHAPE):
    preprocess_input = tf.keras.applications.vgg19.preprocess_input  #προεπεξεργασία των δεδομένων
    base_model = tf.keras.applications.vgg19.VGG19(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
    return base_model, preprocess_input

IMG_SHAPE = IMG_SIZE + (3,)

#Επιλογές των μοντέλων
models = ['resnet50', 'resnet50v2', 'inceptionResnetv2', 'mobilenetv2', 'inceptionv3', 'xception', 'vgg16', 'vgg19']
results0= []
results = []

for n in models:
    temp0 = []
    temp = []
    if n == 'resnet50':
        base_model, preprocess_input = resnet50(IMG_SHAPE)
        fine_tune_at = 600
    elif n == 'resnet50v2':
        base_model, preprocess_input = resnet50v2(IMG_SHAPE)
        fine_tune_at = 600
    elif n == 'inceptionResnetv2':
        base_model, preprocess_input = inceptionResnetv2(IMG_SHAPE) 
        fine_tune_at = 600
    elif n == 'mobilenetv2':
        base_model, preprocess_input = mobilenetv2(IMG_SHAPE) 
        fine_tune_at = 100
    elif n == 'inceptionv3':
        base_model, preprocess_input = inceptionv3(IMG_SHAPE) 
        fine_tune_at = 250      
    elif n == 'xception':
        base_model, preprocess_input = xception(IMG_SHAPE) 
        fine_tune_at = 100
    elif n == 'vgg16':
        base_model, preprocess_input = vgg16(IMG_SHAPE) 
        fine_tune_at = 15
    elif n == 'vgg19':
        base_model, preprocess_input = vgg19(IMG_SHAPE) 
        fine_tune_at = 15

    print('Current model: '+n)
    temp0.append(n)
    temp.append(n)
    image_batch, label_batch = next(iter(trainDS))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)
#Ανάληψη παραμέτρων του μοντέλου. Χρειάζεται να παγώσουν τα στρώματα για να μην 
#γίνει εκπαίδευση, αλλά μόνο validation 

    base_model.trainable = False
#base_model.summary()

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

#Στρώμα prediction για την πρόβλεψη  
    prediction_layer = tf.keras.layers.Dense(38, activation="softmax")
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)


#Δημιουργία μοντέλου μαζί με παραμέτρους και του βασικού προεκπαιδευόμενου μοντέλου 
    inputs = tf.keras.Input(shape=(150, 150, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

#Validation και training του μοντέλου στο καινούργιο Data
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.summary()

    epochs = 10

    history = model.fit(trainDS,
                        epochs=epochs,
                        validation_data=validateDS)

    his = history.history['accuracy'][-1]
    temp0.append(his)
    loss0, accuracy0 = model.evaluate(testDS)
    print(loss0,accuracy0)
    temp0.append(accuracy0)

#Fine tuning
#Το μοντέλο να ανταποκρυθεί καλύτερα στο Dataset χρειάζεται να αλλαχτούν
# τα βάρη με την εκπαίδευση 
#Ξεπαγώνουμε τα στρώματα

    base_model.trainable = True
    print("Number of layers in the base model: ", len(base_model.layers))

#Πρέπει να παγώσουν τα πρώτα fine_tune_at layers απο εκπαίδευση 
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
                metrics=['accuracy'])

    fine_tune_epochs = 10
    epochs =  epochs + fine_tune_epochs

    history_fine = model.fit(trainDS,
                            epochs=epochs,
                            initial_epoch=history.epoch[-1], #ξεκινάει ήδη από το 11 epoch
                            validation_data=validateDS)

    hisf = history_fine.history['accuracy'][-1]
    temp.append(hisf)
    loss, accuracy = model.evaluate(testDS)
    print('Test accuracy :', accuracy)
    temp.append(accuracy)
    results0.append(temp0)
    results.append(temp)
    
    model.save('Model/'+n+'.h5')
print('Initial results')
print(results0)
print('Fine-tuned results')
print(results)
print("---elapsed time %s seconds ---" % (time.time() - start_time))


