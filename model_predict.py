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

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
      raise Exception(e)


def load_model():
    try:
        model_path=os.path.join("artifacts","model.dill")
        # preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
        model=load_object(file_path=model_path)
        return model
    
    except Exception as e:
        raise Exception(e)
    


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
  plt.axis('off')
  plt.imshow(img)
  plt.show()

  # annotate prediction
  bbox = cv_coords(pred)
  start_point = (bbox[0],bbox[1])
  end_point = (bbox[2],bbox[3])
  img=img.copy()
  img = cv2.rectangle(img,start_point,end_point,(0,0,255), 3)
  plt.axis('off')
  plt.imshow(img)
  plt.show()

def predict_selected_image():
  # set index which will be used for prediction and plot
  X_train, y_train_class, y_train_box, X_test, y_test_class, y_test_box = load_model_and_data()
  model = load_model()
  predict(X_test[0], model)
  eval_res = model.evaluate(X_test, {"class_output":y_test_class, "box_output":y_test_box})
  print(eval_res)
# print(f'''Total Loss: {eval_res[0]}\nBBox MSE Loss{eval_res[1]}\nClass BCE Loss: {eval_res[2]}\nBBox MSE: {eval_res[3]}\nAccuracy Score: {eval_res[4]}\n''')
  index = 500

  label = y_test_box[index]

  img = X_test[index]
  pred_box, class_prob = predict(img, model)

  print("Ground Truth: ",label)
  print("Predicted Bbox :", pred_box)
  print("Class  :", class_prob)
  return img,label,pred_box
# plot_prediction(img, label, pred_box)

if __name__ == "__main__":
  img,label,pred_box = predict_selected_image()
  plot_prediction(img, label, pred_box)
   