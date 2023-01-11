#%%
#1. Import packages
from tensorflow.keras import layers, optimizers, losses, metrics, callbacks, applications
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import splitfolders
import numpy as np
import pickle
import os
#%%
#2. Load the data

DATASET_PATH = os.path.join('Concrete Crack Images for Classification')
#%%
#setup hyperparameters for splitfolder
BATCH_SIZE = 32
IMG_SIZE = (48, 48)
SEED1 = 12345

splitfolders.ratio(DATASET_PATH,output="output_img_dataset" ,
    seed=SEED1, ratio=(.7, .2, .1), group_prefix=None, move=False) # default values
#%%
PATH = os.path.join('output_img_dataset')
#%%
#3. Data preparation
#(A) Define the path to the train and validation data folder
train_path = os.path.join(PATH,'train')
val_path = os.path.join(PATH,'val')
test_path = os.path.join(PATH,'test')

#(B) Define the batch size and image size
BATCH_SIZE = 32
IMG_SIZE = (48,48)


#(C) Load the data into tensorflow dataset using the specific method
train_dataset = keras.utils.image_dataset_from_directory(train_path,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)
val_dataset = keras.utils.image_dataset_from_directory(val_path,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)
test_dataset = keras.utils.image_dataset_from_directory(test_path,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)


#%%
#4. Display some images as example
class_names = train_dataset.class_names

plt.figure(figsize=(5,5))
for images,labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
#%%
#5. Further split the validation dataset into validation-test split
#no need split because already split in step #2
"""
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)
"""
#%%
#6. Convert the BatchDataset into PrefetchDataset
AUTOTUNE = tf.data.AUTOTUNE

pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
pf_val = val_dataset.prefetch(buffer_size=AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size=AUTOTUNE)

#%%
#7. Create a small pipeline for data augmentation
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))
#%%
#Apply the data augmentation to test it out
for images,labels in pf_train.take(1):
    first_image = images[0]
    plt.figure(figsize=(5,5))
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image,axis=0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')
#%%
#8. Prepare the layer for data preprocessing
preprocess_input = applications.mobilenet_v2.preprocess_input

#9. Apply transfer learning
IMG_SHAPE = IMG_SIZE + (3,)
feature_extractor = applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

#Disable the training for the feature extractor (freeze the layers)
feature_extractor.trainable = False
feature_extractor.summary()
keras.utils.plot_model(feature_extractor,show_shapes=True)
#%%
#10. Create classification layers
global_avg = layers.GlobalAveragePooling2D()
output_layer = layers.Dense(len(class_names),activation='softmax')

#%%
#11. Use functional API to link all of the modules together

inputs = keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = feature_extractor(x)
x = global_avg(x)
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)


model = keras.Model(inputs=inputs,outputs=outputs)
model.summary()

#%%
#12. Compile the model
optimizer = optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

#%%
#Evaluate the model before model training
loss0,accuracy0 = model.evaluate(pf_test)
print("Loss = ",loss0)
print("Accuracy = ",accuracy0)
#%%
#TensorBoardLog
import datetime
log_path = os.path.join('log_dir','asses3',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = callbacks.TensorBoard(log_dir=log_path)
#%%
#Train the model
EPOCHS = 10
history = model.fit(pf_train,validation_data=pf_val,epochs=EPOCHS,callbacks=[tb])
#%%
"""
Next, we are going to further fine tune the model by using a different transfer learning strategy --> fine tune pretrained model and frozen layers

What we are going to do is we are going to unfreeze some layers at the behind part of the feature extractor, so that those layers will be trained to extract the high features that we specifically want

"""
"""
#13. Apply the next transfer learning strategy
feature_extractor.trainable = True

#Freeze the earlier layers
for layer in feature_extractor.layers[:100]:
    layer.trainable = False

feature_extractor.summary()
#%%
#14. Compile the model
optimizer = optimizers.RMSprop(learning_rate=0.00001)
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

#%%
#15. Continue the model training with this new set of configuration

fine_tune_epoch = 10
total_epoch = EPOCHS + fine_tune_epoch

#Follow up from the previous model training
history = model.fit(pf_train,validation_data=pf_val,epochs=EPOCHS,callbacks=[tb])
"""
#%%
#16. Evaluate the final model
test_loss,test_acc = model.evaluate(pf_test)

print("Loss = ",test_loss)
print("Accuracy = ",test_acc)
#%%
#Deploy the model using the test data
image_batch, label_batch = pf_test.as_numpy_iterator().next()
predictions = np.argmax(model.predict(image_batch),axis=1)

#Compare label and prediction
label_vs_prediction = np.transpose(np.vstack((label_batch,predictions)))
#%%
#Compare label and prediction
label_vs_prediction = np.transpose(np.vstack((label_batch,predictions)))
#%%
#make predictions
image_batch, label_batch = pf_test.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)
class_predictions = np.argmax(predictions,axis=1)
#%%
#show some predictions
plt.figure(figsize=(25,25))

for i in range(16):
    axs = plt.subplot(4,4,i+1)
    plt.imshow(image_batch[i].astype('uint8'))
    current_prediction = class_names[class_predictions[i]]
    current_label = class_names[label_batch[i]]
    plt.title(f"Prediction: {current_prediction}, Actual: {current_label}")
    plt.axis('off')
    
save_path = r"/content"
plt.savefig(os.path.join(save_path,"result.png"),bbox_inches='tight')
plt.show()
     
#%%
#Model Saving
# to save trained model
model.save("model.h5")

# to save one hot encoder model

with open ("ohe.pkl",'wb') as f:
  pickle.dump(ohe,f)
#%%