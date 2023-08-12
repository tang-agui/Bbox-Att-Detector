import random
import torch
import numpy as np
from darknet_v3 import Darknet
import utils
from torchvision import transforms
# import matplotlib.pyplot as plt
# from PIL import Image

population_size = 5
ins_dim_sub = 30  
generations = 100
IMG_DIM = 608
F = 0.5
CR = 0.6
xmin = 0.
xmax = 1.
OBJ_CONF = 0.4  


cfgfile = "cfg/yolov3-dota.cfg"
weightfile = "yolov3-dota_110000.weights"

model = Darknet(cfgfile)

model.load_darknet_weights(weightfile)
model = model.eval().cuda()


def init_population():

    population = []  
    for i in range(population_size):
        rand_value = np.random.rand(3, ins_dim_sub, ins_dim_sub)
        population_j = xmin + rand_value * (xmax-xmin)

        population.append(population_j)
    return population


def calculate_fitness(target_image, population, tru_labels):
    """
    """
    fitness = []
    model.eval()
    for b in range(population_size):
        temp = np.ones((int(ins_dim_full/ins_dim_sub), int(ins_dim_full/ins_dim_sub)))
        # population_b = np.kron(population[b],temp)
        population_b = np.kron(temp, population[b])   
        # single_patch = transforms.ToPILImage(
        #     'RGB')(torch.from_numpy(population[b]).cpu())
        # single_patch.save('target_set/instan_patch/patch_test_save/patch_save_numpy/kron/patch.png')
        # kron_patch = transforms.ToPILImage(
        #     'RGB')(torch.from_numpy(population_b).cpu())
        # kron_patch.save('target_set/instan_patch/patch_test_save/patch_save_numpy/kron/kron_patch.png')

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
        # attack_image = target_image + population_iter_tensor  

        attack_image_instan = transforms.ToPILImage(
            'RGB')(population_iter_tensor.cpu())

        # save_name_instan = 'ship_attack_100/p_30_kron_multi_fitness/patches_save/inst_patch.png'
        # attack_image_instan.save(save_name_instan)
        # # attack_image = torch.from_numpy(attack_image)
        attack_image.clamp_(0, 1)
        attack_image = transforms.ToPILImage('RGB')(attack_image.cpu())
        # save_name_add = 'ship_attack_100/p_30_kron_multi_fitness/patches_save/ins_patch_add.png'
        # attack_image.save(save_name_add)
 
        outputs_boxes = utils.do_detect(
            model, attack_image, 0.4, 0.4, True)

        boxes_attack = []
        for box in outputs_boxes:
            cls_id = box[6]
            if (cls_id == 6):
                if (box[2] >= 0.1 and box[3] >= 0.1):
                    boxes_attack.append(box)
        f_score = 0.0
        
        if len(boxes_attack) == 0:
            fitness.append([-1])
        else:
            outputs_obj_conf = torch.Tensor(boxes_attack)
            all_obj_conf = outputs_obj_conf[:, 4]
            # obj_conf_max = max(all_obj_conf)
            # f_score = obj_conf_max - OBJ_CONF 
            f_score = all_obj_conf - OBJ_CONF
            fitness.append(f_score)
        
    return fitness


def mutation(population):
    #   population = np.zeros((population_size, dim[0],dim[1],dim[2]))
    Mpopulation = np.zeros((population_size, 3, ins_dim_sub, ins_dim_sub))

    for i in range(population_size):
        r1 = r2 = r3 = 0
        F_temp = random.random()  
        if F_temp > 0.5:
            F = 2
        else:
            F = 0.5

        while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2:
            r1 = random.randint(0, population_size - 1)
            r2 = random.randint(0, population_size - 1)
            r3 = random.randint(0, population_size - 1)
        Mpopulation[i] = population[r1] + F * (population[r2] - population[r3])
        '''
        for j in range(dim):
            if xmin <= Mpopulation[i, j] <= xmax:
                Mpopulation[i, j] = Mpopulation[i, j]
            else:
                Mpopulation[i, j] = xmin + random.random() * (xmax - xmin)
        '''
        rand_value = np.random.rand(3, ins_dim_sub, ins_dim_sub)
        population_i = xmin + rand_value * (xmax-xmin)

        Mpopulation[i] = np.where((np.logical_and(
            Mpopulation[i] >= xmin, Mpopulation[i] <= xmax)), Mpopulation[i], population_i)

    return Mpopulation


def crossover(Mpopulation, population):
    #   dim = population.shape  #
    Cpopulation = np.zeros((population_size, 3, ins_dim_sub, ins_dim_sub))
    for i in range(population_size):
        rand_value = np.random.rand(3, ins_dim_sub, ins_dim_sub)
        Cpopulation[i] = np.where(
            rand_value < CR, Mpopulation[i], population[i])
        # Cpopulation[i] = 0.5 * Mpopulation[i] + 0.5 * population[i] 
    '''
    Cpopulation = np.zeros((population_size, dim))
    for i in range(population_size):
        for j in range(dim):
            rand_float = random.random()
            if rand_float <= CR:
                Cpopulation[i, j] = Mpopulation[i, j]
            else:
                Cpopulation[i, j] = population[i, j]
    '''
    return Cpopulation


def selection(taget_image, Cpopulation, population, pfitness, tru_label):

    Cfitness = calculate_fitness(taget_image, Cpopulation, tru_label) 
    for i in range(population_size):
        if len(Cfitness[i]) <= len(pfitness[i]):        #   Cfitness[i] < pfitness[i]:
            population[i] = Cpopulation[i]
            pfitness[i] = Cfitness[i]
        else:
            population[i] = population[i]
            pfitness[i] = pfitness[i]
    return population, pfitness

def fitness_selection(fitness):

    fitness_len = []
    for items in fitness:
        if (len(items) == 1 and items[0] == -1):
            fitness_min_value = -1
            fitness_len.append(len(items))
        else:
            fitness_len.append(len(items))
    fitness_min_len = min(fitness_len)
    
    selected_index = [i for i, x in enumerate(
        fitness_len) if x == fitness_min_len] 
    
    select_list = []
    for i in range(len(selected_index)):
        select_list.append(max(fitness[selected_index[i]]))
    fitness_index = selected_index[np.argmin(select_list)]  
    fitness_index_value = fitness[fitness_index]
    fitness_min_value = max(fitness_index_value)
    return fitness_index, fitness_min_value


def FDE(clean_image, tru_label):

    population = init_population()   

    fitness = calculate_fitness(
        clean_image, population, tru_label) 
    fitness_index, fitness_min_value = fitness_selection(fitness)
    Best_indi = population[fitness_index]
    
    for step in range(generations):
        if fitness_min_value < 0:
            print("break step : ", step)
            break
        Mpopulation = mutation(population)
        Cpopulation = crossover(Mpopulation, population)
        print("step : ", step, "min fitness : ", fitness_min_value)
   
        population, fitness = selection(
            clean_image, Cpopulation, population, fitness, tru_label)
        fitness_index, fitness_min_value = fitness_selection(fitness)
        Best_indi = population[fitness_index]
    # np.save("target_set/npy_data_save/" +
    #         "fitness.npy", fitness_min)
    # plt.plot(fitness_min)
    # plt.savefig("fitness curve.png")
    Best_indi_tensor = torch.from_numpy(Best_indi)
    popu_mask_zeros = np.zeros_like(clean_image)
    # clean_image = clean_image.cpu().detach().numpy()

    temp = np.ones((int(ins_dim_full/ins_dim_sub), int(ins_dim_full/ins_dim_sub)))
    population_best = np.kron(temp, Best_indi_tensor)   
    # population_best = np.zeros((3, ins_dim_full, ins_dim_full))
    # scale_factor = ins_dim_full // ins_dim_sub
    # # temp_b = population[b]

    # for i in range(ins_dim_sub):
    #     for j in range(ins_dim_sub):
    #         population_best[:, scale_factor*i:scale_factor*(
    #             i+1), scale_factor*j:scale_factor*(j+1)] = Best_indi[:, i, j].reshape(3, 1, 1)

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
    # final_image = clean_image + popu_mask_zeros   #
    # final_image = torch.from_numpy(final_image)
    final_image = final_image.float()
    final_image.clamp_(0, 1)

    return final_image


if __name__ == '__main__':
    images = torch.randn(3, 608, 608)

    images = FDE(images, 1)
