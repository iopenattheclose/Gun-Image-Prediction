import tensorflow as tf
from preprocess1 import *
from model_train import *

def predict(input_img_arr, model ,mode='single'):
  if mode is 'single':
    input_img_arr = np.expand_dims(input_img_arr, axis = 0)
    pred_box, class_prob = model.predict(input_img_arr)
    return pred_box[0], class_prob[0]

  else:
    pred_box, class_prob = model.predict(input_img_arr)
    return pred_box, class_prob
  

X_train, X_test, y_train_class, y_test_class, y_train_box, y_test_box,history = startTrain()
predict(X_test[0], ResNet101_final)
eval_res = ResNet101_final.evaluate(X_test, {"class_output":y_test_class, "box_output":y_test_box})
print(f'''Total Loss: {eval_res[0]}\nBBox MSE Loss{eval_res[1]}\nClass BCE Loss: {eval_res[2]}\nBBox MSE: {eval_res[3]}\nAccuracy Score: {eval_res[4]}\n''')