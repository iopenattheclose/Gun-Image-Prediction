import tensorflow as tf
from preprocess1 import *

log_dir = "pistol_Log"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

ResNet101 = tf.keras.applications.ResNet101(weights='imagenet',include_top=False,
                            input_tensor=tf.keras.layers.Input(shape=(416,416,3)))

ResNet101.trainable = False             # make trainable parameter as false

# add some trainable Dense layers
res_output = ResNet101.output
flat = tf.keras.layers.Flatten()(res_output)

# Classification head
x1 = tf.keras.layers.Dense(128,activation='relu')(flat)
x1 = tf.keras.layers.Dense(64,activation='relu')(x1)
x1 = tf.keras.layers.Dense(32,activation='relu')(x1)

# classification output: single class
clas_out = tf.keras.layers.Dense(1,activation='sigmoid',name='class_output')(x1)

# Regression head
x1 = tf.keras.layers.Dense(128,activation='relu')(flat)
x1 = tf.keras.layers.Dense(64,activation='relu')(x1)
x1 = tf.keras.layers.Dense(32,activation='relu')(x1)

# Regression output: 4 B.B Co-ordinates
reg_out = tf.keras.layers.Dense(4,activation='sigmoid', name='box_output')(x1)

ResNet101_final = tf.keras.models.Model(ResNet101.input,[reg_out,clas_out])
print(ResNet101_final.summary())

print("Loss calculation started","\n")

# loss for Classification + Regression
losses = { "class_output":"binary_crossentropy",
          "box_output":"mean_squared_error"}

# loss weight: Optional
loss_weights = {"box_output":4.0,
          "class_output":1.0}

# metrics for both head to track
metrics = {"box_output":"mse",
          "class_output":"accuracy"}

# Optimizer
opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

print("Loss calculation ended","\n")

# Compile model
ResNet101_final.compile(optimizer=opt,loss=losses,loss_weights=loss_weights,metrics=metrics)

def startTrain():
   X_train, X_test, y_train_class, y_test_class, y_train_box, y_test_box = r()
   history = ResNet101_final.fit(x=X_train, y={"class_output":y_train_class, "box_output":y_train_box},
                                 validation_data=(X_test, {"class_output":y_test_class, "box_output":y_test_box}),
                                 batch_size=32,
                                 epochs=10,
                                 callbacks=(tensorboard_callback)
                              )

if __name__ == "__main__":
    startTrain()