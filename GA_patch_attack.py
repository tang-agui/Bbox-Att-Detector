import copy
import random
import torch
import numpy as np
from darknet_v3 import Darknet
import utils
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

population_size = 4
ins_dim_sub = 6  
ins_dim_full = 30
generations = 100
IMG_DIM = 608
###########################################
#   parameters setting
F = 0.01
CR = 0.8
###########################################
xmin = 0.
xmax = 1.
OBJ_CONF = 0.4  # obj condifence threshold


cfgfile = "cfg/yolov3-dota.cfg"
weightfile = "/home/tangguijian/yolov3-dota_110000.weights"

model = Darknet(cfgfile)

model.load_darknet_weights(weightfile)
model = model.eval().cuda()


#   population initialization
def init_population():

    population = np.random.rand(population_size, 3, ins_dim_sub, ins_dim_sub)
    return population

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

        #   patch applier
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
        
        #   instance selection
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

# mutation
def mutation(Cpopulation):

    Mpopulation = copy.deepcopy(Cpopulation)
    for i in range(population_size):

        rand_value = np.random.rand(3, ins_dim_sub, ins_dim_sub)
        population_i = xmin + rand_value * (xmax-xmin)
        
        Mpopulation[i] = np.where(rand_value <F, population_i, Mpopulation[i])
    return Mpopulation

#   crossover
def crossover(population):

    Cpopulation = copy.deepcopy(population)
    pop1 = population[0:population_size //2]
    pop2 = population[population_size//2:]
    
    for i in range(population_size // 2):      
        rand_value = np.random.rand(3, ins_dim_sub, ins_dim_sub)
        Cpopulation[i] = np.where(rand_value < CR, pop2[i],pop1[1])
        Cpopulation[i+population_size//2] = np.where(rand_value < CR, pop1[i], pop2[i])

    return Cpopulation

#   individual selection
def selection(taget_image, Mpopulation, population, pfitness, tru_label):
    Cfitness = calculate_fitness(taget_image, Mpopulation, tru_label)  
    for i in range(population_size):
        if Cfitness[i] < pfitness[i]:
            population[i] = Mpopulation[i]
            pfitness[i] = Cfitness[i]
        else:
            population[i] = population[i]
            pfitness[i] = pfitness[i]
    return population, pfitness



def FDE(clean_image, tru_label):

    population = init_population()  
    fitness = calculate_fitness(
        clean_image, population, tru_label)  
    Best_indi_index = np.argmin(fitness)    
    Best_indi = population[Best_indi_index]
    fitness_min = []
    for step in range(generations):
        if min(fitness) < 0:
            print("break step : ", step)
            break

        fit_min = min(fitness)
        fitness_min.append(fit_min)
        Cpopulation = crossover(population)
        Mpopulation = mutation(Cpopulation)
        #   monitor
        print("step : ", step, "min fitness : ", fit_min)
        
        population, fitness = selection(
            clean_image, Mpopulation, population, fitness, tru_label)
        Best_indi_index = np.argmin(fitness)
        Best_indi = population[Best_indi_index]

    Best_indi_tensor = torch.from_numpy(Best_indi)
    popu_mask_zeros = np.zeros_like(clean_image)

    temp = np.ones((int(ins_dim_full/ins_dim_sub),
                   int(ins_dim_full/ins_dim_sub)))
    population_best = np.kron(temp, Best_indi_tensor) 

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
