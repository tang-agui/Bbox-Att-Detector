
from PIL import Image
import PIL.Image
import PIL.ImageOps
from utils import *
from darknet_v3 import Darknet
import numpy as np
import matplotlib.pyplot as plt


def eval_list(pre_patched_labels_dir, labdir, conf_thresh, iou_thresh):

    conf_thresh = conf_thresh  
    iou_thresh = iou_thresh  
    total = 0.0
    proposals = 0.0
    correct = 0.0
    lineId = 0
    avg_iou = 0.0
    
    total_ASR = 0. 

    for imgfile in os.listdir(pre_patched_labels_dir): 
        # print("new image")
        if imgfile.endswith('.txt'):
            pre_labeld_dir = os.path.abspath(os.path.join(pre_patched_labels_dir, imgfile))
            lab_path = os.path.abspath(os.path.join(
                labdir, imgfile))
            lineId = lineId + 1 

            truths = read_truths_pre_7(lab_path)  
            boxes_cls = read_truths_pre_7(pre_labeld_dir)
            total_ASR += len(boxes_cls)     

            total = total + truths.shape[0]  # ground_truth labels
            # print("length of boxes_cls : ", len(boxes_cls))
            for i in range(len(boxes_cls)):
                if (boxes_cls[i][4]) > conf_thresh:  
                    proposals = proposals+1  # 
            for i in range(truths.shape[0]):
                box_gt = [truths[i][1], truths[i][2],
                          truths[i][3], truths[i][4], 1.0]
                best_iou = 0
                for j in range(len(boxes_cls)):
                    iou = bbox_iou(
                        box_gt, boxes_cls[j], x1y1x2y2=False)  # 
                    best_iou = max(iou, best_iou)  # 
                if best_iou > iou_thresh:
                    avg_iou += best_iou
                    correct = correct+1  
    ASR = (total - total_ASR) / total
    precision = correct/(proposals + 1e-8)  
    recall = correct/(total + 1e-8)  
    fscore = 2.0*precision*recall/(precision+recall + 1e-6)
    # print("results in recall.py :")
    # print("%d IOU: %f, Recal: %f, Precision: %f, Fscore: %f\n" %
    #       (lineId-1, avg_iou/(correct + 1e-6), recall, precision, fscore))
    # print("total images = ", n_images)
    return precision, recall, ASR




def exif_transpose(img):
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img


def load_image_file(file, mode='RGB'):
    # Load the image with PIL
    img = PIL.Image.open(file)

    if hasattr(PIL.ImageOps, 'exif_transpose'):
        # Very recent versions of PIL can do exit transpose internally
        img = PIL.ImageOps.exif_transpose(img)
    else:
        # Otherwise, do the exif transpose ourselves
        img = exif_transpose(img)

    img = img.convert(mode)

    return img

def txt_len_read(txtfile_list):
    # for instances calculate
    len_txt = 0
    len_ins_account = []
    for txtfile_label in os.listdir(txtfile_list):  
        txtfile = os.path.abspath(os.path.join(txtfile_list, txtfile_label)) 
        if os.path.getsize(txtfile):
            myfile = open(txtfile)
            single_len = len(myfile.readlines())
            len_txt += single_len
            len_ins_account.append(single_len)

    return len_txt #, len_ins_account


from torchvision import transforms
from PIL import Image
import torch



def img_transfer(img):
    
    if isinstance(img, Image.Image):
      
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))

        img = img.view(height, width, 3).transpose(
            0, 1).transpose(0, 2).contiguous()  
        img = img.view(3, height, width)
        img = img.float().div(255.0)  #
    elif type(img) == np.ndarray:  #
        img = torch.from_numpy(img.transpose(
            2, 0, 1)).float().div(255.0)
    else:
        print("unknown image type")
        exit(-1)
    return img
