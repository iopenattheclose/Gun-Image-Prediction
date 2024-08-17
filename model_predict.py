import tensorflow as tf
from preprocess1 import *
from model_train import *
import matplotlib.pyplot as plt
import cv2

img_h = 416
img_w = 416

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

def cv_coords(box):
  '''
  This function will convert B.B cordinates from (X_center,Y_center,width,height)  format
  to (x_start,y_start,x_end,y_end) required for cv2 to create rectangle
  '''
  x,y,w,h= box[0],box[1],box[2],box[3]
  x1, y1 = int((x-w/2)*img_w), int((y-h/2)*img_h)
  x2, y2 = int((x+w/2)*img_w), int((y+h/2)*img_h)
  return x1, y1, x2, y2



def plot_prediction(img, label, pred):
  img = img.copy()
  plt.figure(figsize=(8,8))
  print("ImgResolution",img.shape)

  # annotate ground truth
  bbox = cv_coords(label)
  start_point = (bbox[0],bbox[1])
  end_point = (bbox[2],bbox[3])
  img=img.copy()
  img = cv2.rectangle(img,start_point,end_point,(255,255,0), 3)
  cv2.putText(img, 'Ground Truth', (start_point[0],start_point[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))


  # annotate prediction
  bbox = cv_coords(pred)
  start_point = (bbox[0],bbox[1])
  end_point = (bbox[2],bbox[3])
  img=img.copy()
  img = cv2.rectangle(img,start_point,end_point,(0,0,255), 3)
  cv2.putText(img, 'Prediction', (start_point[0],start_point[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))


  plt.axis('off')
  plt.imshow(img)

# set index which will be used for prediction and plot
index = 500

label = y_test_box[index]

img = X_test[index]
pred_box, class_prob = predict(img, ResNet101_final)

print("Ground Truth: ",label )
print("Predicted Bbox :", pred_box)

plot_prediction(img, label, pred_box)