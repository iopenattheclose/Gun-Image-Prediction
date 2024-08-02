import matplotlib.pyplot as plt
import cv2
import filepath as fp
# from explore import * -> no need to call method like ex.getImageFolderDetails -> call method directly

img_h = 416
img_w = 416

def cv_coords(box):
  '''
  This function will convert B.B cordinates from (X_center,Y_center,width,height) format
  to (x_start,y_start,x_end,y_end) required for cv2 to annotate rectangle on objects
  '''
  x,y,w,h= box[1],box[2],box[3],box[4]
  x1, y1 = int((x-w/2)*img_w), int((y-h/2)*img_h)
  x2, y2 = int((x+w/2)*img_w), int((y+h/2)*img_h)
  return x1, y1, x2, y2


def plot_img_and_box(img, bbox):
  plt.figure(figsize=(6,6))
  print("ImgResolution",img.shape)
  bbox = cv_coords(bbox)
  start_point = (bbox[0],bbox[1])
  end_point = (bbox[2],bbox[3])
  img=img.copy()
  img = cv2.rectangle(img,start_point,end_point,(255,255,0), 3)
  plt.axis('off')
  plt.imshow(img)
  plt.show()

for i in range(5):
  datapath, annot_files, img_files = fp.getImageFolderDetails()
  img_path = f'{datapath}pistol_images/{i+1}.jpg'
  label_path = f'{datapath}pistol_annotations/{i+1}.txt'


  img = plt.imread(img_path)

  with open(label_path,'r') as f:
    bbox = (f.readlines())
    bbox = [float(element) for element in bbox[0].split(" ")]
    print(bbox)
  plot_img_and_box(img, bbox)