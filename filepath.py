import os
import matplotlib.pyplot as plt
import cv2


def getImageFolderDetails():
    datapath = '/Users/prupro/Desktop/Github/PistolData_merged/'
    annot_files = sorted(os.listdir(f'{datapath}pistol_annotations'))
    img_files = sorted(os.listdir(f'{datapath}pistol_images'))
    return datapath, annot_files, img_files

def printFileDetails():
    datapath, annot_files, img_files = getImageFolderDetails()
    print(f'Total files: {len(annot_files)}')
    print(f'Top 5 annotation files: {annot_files[:5]}')
    print(f'Top 5 image files: {img_files[:5]}')


def printAnnotationFileDetails():
    datapath, annot_files, img_files = getImageFolderDetails()
    for annot_file in annot_files[0:5]:
        annot_file_path = os.path.join(datapath, 'pistol_annotations', annot_file)
        with open(annot_file_path, 'r') as f:
            print(f.readlines())
