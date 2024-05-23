import copy
import random
import torch
import numpy as np
from darknet_v3 import Darknet
import utils
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

population_size = 5
ins_dim_sub = 30  
ins_dim_full = 30
generations = 100
IMG_DIM = 608
###########################################
#   hyper-parameters 
var_max = 1.
var_min = 0.
Vmax =0.5 * (var_max - var_min)
Vmin = -Vmax
omega = 0.5
##################################

c1 = 2.0   
c2 = 2.0
max_fes = 500
OBJ_CONF = 0.4  

cfgfile = "cfg/yolov3-dota.cfg"
weightfile = "/home/tangguijian/yolov3-dota_110000.weights"
model = Darknet(cfgfile)
model.load_darknet_weights(weightfile)
model = model.eval().cuda()

#   population initialization
def init_population():
    pos_rand_value = np.random.rand(population_size, 3, ins_dim_sub, ins_dim_sub)
    vel_rand_value = np.random.rand(population_size, 3, ins_dim_sub, ins_dim_sub)
    pos = var_min + pos_rand_value*(var_max-var_min)
    vel = Vmin + 2 * Vmax *vel_rand_value
    return pos, vel

#   fitness value calculation
def calculate_fitness(target_image, population, tru_labels):
    """
    """
    fitness = []
    model.eval()
    for b in range(population_size):
        temp = np.ones((int(ins_dim_full/ins_dim_sub), int(ins_dim_full/ins_dim_sub)))
        population_b = np.kron(temp, population[b])   
        popu_mask_zeros = np.zeros_like(target_image)

        for j in range(len(tru_labels)):
            w_0 = tru_labels[j][0]  # (x,y)
            h_0 = tru_labels[j][1]
            x_0 = int(w_0 * IMG_DIM) 
            y_0 = int(h_0 * IMG_DIM)  
            popu_mask_zeros[:, y_0-int(ins_dim_full/2):y_0+int(ins_dim_full/2),
                            x_0-int(ins_dim_full/2):x_0+int(ins_dim_full/2)] = population_b
            
        population_iter_tensor = torch.from_numpy(popu_mask_zeros)
        attack_image = torch.where(
            (population_iter_tensor == 0), target_image, population_iter_tensor)

        attack_image.clamp_(0, 1)
        attack_image = transforms.ToPILImage('RGB')(attack_image.cpu())
        outputs_boxes = utils.do_detect(
            model, attack_image, 0.4, 0.4, True)
        
        boxes_attack = []
        for box in outputs_boxes:
            cls_id = box[6]
            if (cls_id == 0):
                boxes_attack.append(box)
        f_score = 0.0
        if len(boxes_attack) == 0:
            fitness.append(-1)
        else:
            outputs_obj_conf = torch.Tensor(boxes_attack)
            all_obj_conf = outputs_obj_conf[:, 4]
            obj_conf_max = max(all_obj_conf)
            f_score = obj_conf_max - OBJ_CONF  
            fitness.append(f_score)

    return fitness

def PSO_AT(clean_image, tru_label):

    fes = 0.
    pop_pos, pop_vel = init_population()  
    ipop = copy.deepcopy(pop_pos)
    pfitness = calculate_fitness(clean_image, ipop, tru_label) 
    fes += population_size
    pbest = copy.deepcopy(pop_pos)
    
    gindex = np.argmin(pfitness)
    gbest = pbest[gindex]  
    
    while fes < max_fes:
        if(min(pfitness) < 0):
            print("break step : ", fes)
            break
        for i in range(population_size):
            pop_vel[i] = omega * pop_vel[i] +c1*np.random.rand(3,ins_dim_sub,ins_dim_sub)*(pbest[i]-pop_pos[i])+c2*np.random.rand(3,ins_dim_sub,ins_dim_sub)*(gbest-pop_pos[i])
            pop_pos[i] = pop_pos[i] + pop_vel[i]
        np.clip(pop_pos, var_min, var_max)
        
        print("step : ", fes, "min fitness : ", min(pfitness))
        offer_pop = copy.deepcopy(pop_pos)
        offer_fitness = calculate_fitness(clean_image, offer_pop, tru_label)
        fes += population_size
        
        for i in range(population_size):
            if pfitness[i] > offer_fitness[i]:
                pfitness[i] = offer_fitness[i]
                pbest[i] = pop_pos[i]

        gindex = np.argmin(pfitness)
        gbest = pbest[gindex]
    
    Best_indi_tensor = torch.from_numpy(gbest)
    popu_mask_zeros = np.zeros_like(clean_image)

    temp = np.ones((int(ins_dim_full/ins_dim_sub),
                   int(ins_dim_full/ins_dim_sub)))
    population_best = np.kron(temp, Best_indi_tensor)  # 

    for j in range(len(tru_label)):
        w_0 = tru_label[j][0]  # (x,y)
        h_0 = tru_label[j][1]
        x_0 = int(w_0 * IMG_DIM)
        y_0 = int(h_0 * IMG_DIM)

        popu_mask_zeros[:, y_0-int(ins_dim_full/2):y_0+int(ins_dim_full/2),
                        x_0-int(ins_dim_full/2):x_0+int(ins_dim_full/2)] = population_best

    final_pertur = torch.from_numpy(popu_mask_zeros)
    final_image = torch.where(
        (final_pertur == 0.), clean_image, final_pertur)

    final_image = final_image.float()
    final_image.clamp_(0, 1)

    return final_image
