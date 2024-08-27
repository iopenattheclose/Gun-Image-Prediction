import numpy as np
from filepath import *
from sklearn.model_selection import train_test_split
import dill

datapath, annot_files, img_files = getImageFolderDetails()
train_data_path: str=os.path.join('artifacts',"train.dill")
test_data_path: str=os.path.join('artifacts',"test.dill")

def return_bbox(label_path):
  ''' read and return bbox from text file'''
  with open(label_path,'r') as f:
    bbox = (f.readlines())
    bbox = [float(element) for element in bbox[0].split(" ")]
    return bbox[0],bbox[1:]

def return_image_label(filename):
  '''
  read and return image as well as corresponding label and bbox from image file
  '''
  try:
    img_path = f'{datapath}pistol_images/{filename}.jpg'
    image = plt.imread(img_path)
    label_path = f'{datapath}pistol_annotations/{filename}.txt'
    class_label,bbox_label = return_bbox(label_path)
    return True,image,class_label,bbox_label
  except Exception as e:
    print("Exception: ",filename, str(e))
    return False,None,None, None


### Looping over all the files present in directory to extract image, labels and bbox
images, class_labels, bbox_labels = [], [],[]
for filename in annot_files:
  filename = filename[:-4]
  status, image, class_label, bbox_label= return_image_label(filename)
  if status==True and image.shape==(416,416,3):
    images.append(image)
    class_labels.append(class_label)
    bbox_labels.append(bbox_label)

def printDetails(X_train, X_test, y_train_class, y_test_class, y_train_box, y_test_box):
  print("Image shape : Num of images,h,w,dimension :",images.shape)
  print("Class labels shape :",class_labels.shape)
  print("Box labels shape :",bbox_labels.shape)
  print("Sample class labels",class_labels[125:140])
  print("Sample bbox labels",bbox_labels[231:245])
  print(np.unique(class_labels, return_counts=True))
  # there are 999 images with no guns while 2704 images with Guns present in frame
  print(X_train.shape, X_test.shape, y_train_class.shape, y_test_class.shape,y_train_box.shape, y_test_box.shape)
  print(y_train_box[:10])
  print(y_train_class[:10])





images,class_labels,bbox_labels = np.array(images),np.array(class_labels),np.array(bbox_labels)

def split_images():
  X_train, X_test, y_train_class, y_test_class, y_train_box, y_test_box  = train_test_split(
      images, class_labels, bbox_labels, test_size=0.30, random_state=42)
  # printDetails(X_train, X_test, y_train_class, y_test_class, y_train_box, y_test_box)
  # return X_train, X_test, y_train_class, y_test_class, y_train_box, y_test_box

  train_data = {
      "X_train": X_train,
      "y_train_class": y_train_class,
      "y_train_box": y_train_box
  }
  save_data(train_data_path,train_data)
  

  # Save test data
  test_data = {
      "X_test": X_test,
      "y_test_class": y_test_class,
      "y_test_box": y_test_box
  }

  save_data(test_data_path,test_data)

def save_data(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise Exception(e)

if __name__ == "__main__":
  split_images()