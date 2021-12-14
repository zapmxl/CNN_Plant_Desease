'''
Αλγόριθμος για Transfer Learning CNN Image Classification
Μοντέλο VGG1`9
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

classes = trainDS.class_names
nClasses =  38

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

#Στρώμα προεπεξεργασίας του data σε μορφη [-1,1] για το μοντέλο MobileNetV2
#rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)

#Άλλος τρόπος προεπεξεργασίας
preprocess_input = tf.keras.applications.resnet_v2.preprocess_input

#Δημιουργία προεκπαιδευόμενων μοντέλων convnets
#Δημιουργία μοντέλου MobileNetV2  (1.4 εκατ. εικόνες και 1000 κλάσεις)
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.resnet_v2.ResNet50V2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

image_batch, label_batch = next(iter(trainDS))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

#Ανάληψη παραμέτρων του μοντέλου. Χρειάζεται να παγώσουν τα στρώματα για να μην 
#γίνει εκπαίδευση, αλλά μόνο validation 
print("Number of layers in the base model: ", len(base_model.layers))
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
temp0 = []
his = history.history['accuracy'][-1]
temp0.append(his)
loss0, accuracy0 = model.evaluate(testDS)
temp0.append(accuracy0)


#Fine tuning
#Το μοντέλο να ανταποκρυθεί καλύτερα στο Dataset χρειάζεται να αλλαχτούν
# τα βάρη με την εκπαίδευση 
#Ξεπαγώνουμε τα στρώματα

base_model.trainable = True

#Πρέπει να παγωθούν τα πρώτα layers απο εκπαίδευση 
fine_tune_at = 160
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
temp = []
hisf = history_fine.history['accuracy'][-1]
temp.append(hisf)
loss, accuracy = model.evaluate(testDS)
temp.append(accuracy)

print('Initial results')
print(temp0)
print('Fine-tuned results')
print(temp)

model.save('Model/resnet50.h5')
print("---elapsed time %s seconds ---" % (time.time() - start_time))
