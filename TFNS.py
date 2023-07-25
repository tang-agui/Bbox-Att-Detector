import time
import numpy as np
import torchvision.transforms as transforms
from TDE_main_ins_patch_multi_fit import FDE
from darknet_v3 import Darknet
import os
import utils_self
import utils
from PIL import Image
import fnmatch
import TDE_main_ins_patch_multi_fit


print('hyper-paramenters,population size : {}, generations : {}, xmin : {:.4f}, xmax : {:.4f}, per ins dim_full : {:.4f}, per ins dim_sub : {:.4f}'.format(
    TDE_main_ins_patch_multi_fit.population_size, TDE_main_ins_patch_multi_fit.generations,
    TDE_main_ins_patch_multi_fit.xmin, TDE_main_ins_patch_multi_fit.xmax, TDE_main_ins_patch_multi_fit.ins_dim_full,
    TDE_main_ins_patch_multi_fit.ins_dim_sub))

print("start training time : ", time.strftime('%Y-%m-%d %H:%M:%S'))

imgdir = '/mnt/share1/tangguijian/Data_storage/Black_AE_Evo_testset/images'
clean_labdir = "/mnt/share1/tangguijian/Data_storage/Black_AE_Evo_testset/yolo-labels"
savedir = "plane_attack_100/kron_multi_fitness_adap_scale"


print("savedir : ", savedir)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#   YOLOv3
cfgfile = "cfg/yolov3-dota.cfg"
weightfile = "/mnt/share1/tangguijian/DOTA_YOLOv3_patch_AT/weights/yolov3-dota_110000.weights"

#   Faster R-CNN

# cfgfile = "cfg/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_dota.py"
# weightfile = "/mnt/jfs/tangguijian/AerialDetection_black/work_dirs/faster_rcnn_RoITrans_r50_fpn_1x_dota/epoch_12.pth"

model = Darknet(cfgfile)

model.load_darknet_weights(weightfile)
model = model.eval().cuda()

count = 0
total_count = 0
net_correct = 0

img_size = model.height
img_width = model.width
class_names = utils.load_class_names('data/dota.names')

instances_clean = []
instances_after_attack = []

t_begin = time.time()

n_png_images = len(fnmatch.filter(
    os.listdir(imgdir), '*.png'))
n_jpg_images = len(fnmatch.filter(
    os.listdir(imgdir), '*.jpg'))
n_images = n_png_images + n_jpg_images  
print("Total images in testset : ", n_images)

for imgfile in os.listdir(imgdir):
    print("new image")  #
    t_single_begin = time.time()
    if imgfile.endswith('.jpg') or imgfile.endswith('.png'):  
        name = os.path.splitext(imgfile)[0]  # image name w/o extension
    
        txtname = name + '.txt'  
        txtpath = os.path.abspath(os.path.join(clean_labdir, txtname))

        imgfile = os.path.abspath(os.path.join(imgdir, imgfile))  # abs path
        print("image file path is ", imgfile)
        img = utils_self.load_image_file(imgfile)  
        w, h = img.size
        # print("original w = ", w, ", h = ", h)
        if w == h:
            padded_img = img
        else:
            dim_to_pad = 1 if w < h else 2   # dim_to_pad = 1 if w < h else 2  
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new(
                    'RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(img, (int(padding), 0))
            else:
                padding = (w - h) / 2
                padded_img = Image.new(
                    'RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(img, (0, int(padding)))

        resize = transforms.Resize((img_size, img_size))
        padded_img = resize(padded_img)
 
        tru_lab = utils.read_truths_pre_7(txtpath)  # array
        # print("tru_lab : ", tru_lab[0], "int : ", int(tru_lab[0][0]*608))
        # _, pre = torch.max(outputs.data, 1)
        total_count += len(tru_lab)  # 
        instances_clean.append(len(tru_lab))  

        if len(tru_lab):  
            net_correct += len(tru_lab)
            images_to_attack = utils_self.img_transfer(
                padded_img)  # tensor,[3x608x608],
            # print("size of : ", img.size())
            
            images_attack = FDE(images_to_attack, tru_lab)
            images_attack = transforms.ToPILImage(
                'RGB')(images_attack.cpu())  # 

            boxes_cls_attack = utils.do_detect(
                model, images_attack, 0.4, 0.4, True)
            boxes_attack = []  # 
            for box in boxes_cls_attack:
                cls_id = box[6]
                if (cls_id == 0):
                    if (box[2] >= 0.1 and box[3] >= 0.1):
                        boxes_attack.append(box)

            count += len(boxes_attack)  
            instances_after_attack.append(
                len(boxes_attack))  
                
            txtpath_write = os.path.abspath(os.path.join(
                savedir, 'yolo-labels/', txtname))
            textfile = open(txtpath_write, 'w+')

            for box in boxes_attack:
                textfile.write(
                    f'{box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]} {box[6]}\n')
            textfile.close()
            print("single image tru-instances : ", len(tru_lab), "instances after attack : ", len(boxes_attack), "instances gap : ", (len(tru_lab)-len(boxes_attack)))
        attacked_count = net_correct - count


    print("image ", imgfile, "attack done!")
    t_single_end = time.time()
    print('singel attack time: {:.4f} minutes'.format(
        (t_single_end - t_single_begin) / 60))
print("total instances : ", total_count, "correct clean instances : ", net_correct,
      "instances after attack : ", count, "total instances gap : ", attacked_count)
if net_correct > 0:
    print('Accuracy of attack: %f %%' %
          (100 * float(attacked_count) / net_correct))


print("All Done!")

t_end = time.time()
print('Total attack time: {:.4f} minutes'.format(
    (t_end - t_begin) / 60))
