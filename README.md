
# Image_classification_concrete_crack 

I. This project is to detected whether the concrete is crack or not

II. The data is from GitHub - https://data.mendeley.com/datasets/5y9wdsg2zt/2

III. this project is carry out with (Problem
Formulation →Data Preparation→Model Development→Model
Deployment)

IV. This project using TensorFlow transfer learning.

V. The accuracy is 99%




## Project Output

125/125 [=====] - 16s 126ms/step - loss: 0.0127 - accuracy: 0.9967

Loss =  0.0126899853348732

Accuracy =  0.996749997138977

## Model Architecture 
![alt text](https://github.com/lemon5969/Image_classification_concrete_crack/blob/main/Image/model.png?raw=true)

##Training
Epoch 1/10
875/875 [==============================] - 200s 216ms/step - loss: 0.1303 - accuracy: 0.9542 - val_loss: 0.0391 - val_accuracy: 0.9898

Epoch 2/10
875/875 [==============================] - 175s 200ms/step - loss: 0.0370 - accuracy: 0.9909 - val_loss: 0.0286 - val_accuracy: 0.9931

Epoch 3/10
875/875 [==============================] - 149s 170ms/step - loss: 0.0291 - accuracy: 0.9920 - val_loss: 0.0223 - val_accuracy: 0.9942

Epoch 4/10
875/875 [==============================] - 160s 182ms/step - loss: 0.0242 - accuracy: 0.9932 - val_loss: 0.0200 - val_accuracy: 0.9946

Epoch 5/10
875/875 [==============================] - 140s 160ms/step - loss: 0.0208 - accuracy: 0.9946 - val_loss: 0.0195 - val_accuracy: 0.9946

Epoch 6/10
875/875 [==============================] - 141s 161ms/step - loss: 0.0210 - accuracy: 0.9945 - val_loss: 0.0171 - val_accuracy: 0.9952

Epoch 7/10
875/875 [==============================] - 141s 161ms/step - loss: 0.0181 - accuracy: 0.9948 - val_loss: 0.0180 - val_accuracy: 0.9949

Epoch 8/10
...
Epoch 9/10
875/875 [==============================] - 172s 196ms/step - loss: 0.0170 - accuracy: 0.9952 - val_loss: 0.0148 - val_accuracy: 0.9961

Epoch 10/10
875/875 [==============================] - 161s 184ms/step - loss: 0.0158 - accuracy: 0.9957 - val_loss: 0.0147 - val_accuracy: 0.9958

## Result Graph
Below is Tensorboard graph training
![alt text](https://github.com/lemon5969/Image_classification_concrete_crack/blob/main/Image/TB.png?raw=true)

Here is Validation output image
![alt text](https://github.com/lemon5969/Image_classification_concrete_crack/blob/main/Image/validation.png?raw=true)

## Conclusion
This model can verify the concrete wheher is crack or not with accuracy 99.6%.


## Credits:
This data sets is taken from https://data.mendeley.com/datasets/5y9wdsg2zt/2
Thank you :)


