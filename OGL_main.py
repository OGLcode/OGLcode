import gc
import os
import sys
import time
import glob
import numpy as np
from numpy import append
from scipy import min_scalar_type
import random
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from torch.autograd import Variable
from model_search import Network


import genotypes
from genotypes import PRIMITIVES

import copy
from default_option import TrainOptions

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("cifar10")
parser.add_argument('--data', type=str, default='../../Cifar/cifar-10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')#######change the epochs
parser.add_argument('--init_channels', type=int, default=52, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.4, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=100, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.7, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10
opt=TrainOptions()

def random_arch_generate():#######randomly generate architecture
    num_ops = len(genotypes.PRIMITIVES)
    n_nodes = 4####model._step

    arch_gene = []
    for i in range(n_nodes):
        ops = np.random.choice(range(num_ops), 2)
        nodes_in_normal = np.random.choice(range(i+2), 2, replace=False)
        arch_gene.extend([(ops[0],nodes_in_normal[0]), (ops[1],nodes_in_normal[1])])
    return arch_gene  

def cal_arch_dis(arch1,arch2):##calculate the distance, smaller distance more similar
    dis=8
    n_nodes=4######genotypes.STEPS

    for i in range(n_nodes):
        if arch1[2*i]==arch2[2*i]:
            dis=dis-1
        elif arch1[2*i]==arch2[2*i+1]:
            dis=dis-1
        if arch1[2*i+1]==arch2[2*i+1]:
            dis=dis-1
        elif arch1[2*i+1]==arch2[2*i]:
            dis=dis-1                      
    dis=dis/8
    return dis 



def cal_diver_score(arch,archive):######KNN based diversity calculation
    n=len(archive)
    dis=np.zeros(n)
    for i in range(n):
        dis[i]=cal_arch_dis(arch,archive[i])
        
    sort_dis=np.sort(dis)
    if len(sort_dis) !=0:
        diver_score=np.mean(sort_dis[0:10])##k=10 for knn
    else:
        diver_score = 0
    return diver_score
 

    
def diver_arch_generate(arch_archive):############randomly genrate architecture and get the best one
    ini_diver_score=0
    arch_g=random_arch_generate()
    for i in range(10):##################repeat 10 times to get the diversified architecture
        arch=random_arch_generate()         
        diver_score=cal_diver_score(arch,arch_archive)#
        if diver_score>ini_diver_score:
            arch_g=arch
            ini_diver_score=diver_score
            
    return arch_g


def diver_arch_replace(index,arch_archive,archive_recent):#######randomly generate architecture to repalce
    arch_compar=arch_archive[index]
    a=np.arange(0,index)
    b=np.arange(index+1,len(arch_archive))
    index_remain=np.append(a,b)
    
    arch_archive_remain=[arch_archive[j] for j in index_remain]
    
    ini_diver_score=cal_diver_score(arch_compar,arch_archive_remain)
    for i in range(len(archive_recent)):##############select diversified architetcure from recent architectures
        arch=archive_recent[i] 
        diver_score=cal_diver_score(arch,arch_archive_remain)
        if diver_score>ini_diver_score:
            arch_compar=arch
            ini_diver_score=diver_score
            
    return arch_compar


def find_similar_arch(arch,archive):#####get the index of the most similar architecture
    dis=np.zeros(len(archive))   
    
    for i in range(len(archive)):
        dis[i]=cal_arch_dis(arch,archive[i])

    m=np.argsort(dis)
    index=m[0]
    
    return index


def arch_archive_update(arch_gene,arch_archive,normal_archive_recent,reduction_archive_recent):#####update architecture archive (also the constraint subset)
    store_num=8###set the ARCIVE number M
    if len(arch_archive)==2*store_num:
        ind_arch_norm_replace=find_similar_arch(arch_gene[0],arch_archive[0:len(arch_archive):2])
        ind_arch_redu_replace=find_similar_arch(arch_gene[1],arch_archive[1:len(arch_archive):2])
        arch_archive[2*ind_arch_norm_replace]=diver_arch_replace(ind_arch_norm_replace,arch_archive[0:len(arch_archive):2],normal_archive_recent)
        arch_archive[2*ind_arch_redu_replace+1]=diver_arch_replace(ind_arch_redu_replace,arch_archive[1:len(arch_archive):2],reduction_archive_recent)
        
    else:
        normal_arch=diver_arch_generate(arch_archive[0:len(arch_archive):2])
        reduce_arch=diver_arch_generate(arch_archive[1:len(arch_archive):2])######greedy
        arch_archive.append(normal_arch)
        arch_archive.append(reduce_arch)
    return arch_archive


def get_weights_from_arch(arch_comb):########get the continuous representation of architecture
    k = sum(1 for i in range(model._steps) for n in range(2+i))
    num_ops = len(genotypes.PRIMITIVES)
    n_nodes = model._steps

    alphas_normal = Variable(torch.zeros(k, num_ops).cuda(), requires_grad=False)
    alphas_reduce = Variable(torch.zeros(k, num_ops).cuda(), requires_grad=False)

    offset = 0
    for i in range(n_nodes):
        normal1 = np.int_(arch_comb[0][2*i])
        normal2 = np.int_(arch_comb[0][2*i+1])
        reduce1 = np.int_(arch_comb[1][2*i])
        reduce2 = np.int_(arch_comb[1][2*i+1])
        alphas_normal[offset+normal1[1],normal1[0]] = 1
        alphas_normal[offset+normal2[1],normal2[0]] = 1
        alphas_reduce[offset+reduce1[1],reduce1[0]] = 1
        alphas_reduce[offset+reduce2[1],reduce2[0]] = 1
        offset += (i+2)

    model_weights = [
      alphas_normal,
      alphas_reduce,
    ]
    return model_weights


def set_model_weights(model, weights):#####set the architecture weights for the supernet
    model.alphas_normal = weights[0]
    model.alphas_reduce = weights[1]
    model._arch_parameters = [model.alphas_normal, model.alphas_reduce]
    return model


def cal_loss_archive(arch_gene,arch_archive_new,model_save,input,target,criterion):###get the mean loss of all constraint architecture
    loss_arch=0
    
    for i in range(np.int(len(arch_archive_new)/2)):
        w1=1-cal_arch_dis(arch_gene[0],arch_archive_new[2*i])
        w2=1-cal_arch_dis(arch_gene[1],arch_archive_new[2*i+1])
        w=(w1+w2)/2
        model_save_save=copy.deepcopy(model_save)        
        model_weights=get_weights_from_arch(arch_archive_new[2*i:2*i+2])  
        model_save_save=set_model_weights(model_save_save,model_weights)
        
        logits = model_save_save(input)        
        loss=criterion(logits, target)
        loss_arch=w*(loss_arch+loss.item())
        del model_save_save
    loss_archive=(loss_arch*2)/len(arch_archive_new)
    del model_save
    return loss_archive

def _parse_D(weights):####get the architectures' number representation with fixed depth
    gene = []
    n = 2
    start = 0
    for i in range(4):
        end = start + n
        W = weights[start:end].copy()
        edges = [0,i+1]
        for j in edges:
            k_best = None
            for k in range(len(W[j])):
                if k != PRIMITIVES.index('none'):
                    if k_best is None or W[j][k] > W[j][k_best]:
                        k_best = k
            gene.append((k_best, j))
        start = end
        n += 1
    return gene

def _parse_gene(weights):####get the architectures' number representation with fixed depth
    gene = []
    n = 2
    start = 0
    for i in range(4):
        end = start + n
        W = weights[start:end].copy()
        edges =  sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
            k_best = None
            for k in range(len(W[j])):
                if k != PRIMITIVES.index('none'):
                    if k_best is None or W[j][k] > W[j][k_best]:
                        k_best = k
            gene.append((k_best, j))
        start = end
        n += 1
    return gene

def arc_update(p, Arc_p_index, w):
    global Arc_p
    F, _, W, H = w.shape
    if W == 1 and H == 1:
        return
    if p.cpu().equal(torch.eye(1)):
        mem = torch.Tensor(w.grad.data.contiguous().view(F, -1))
    else:
        mem = torch.cat((p, w.grad.data.contiguous().view(F, -1)), axis=0)

    # if mem.size(0) > mem.size(1):
    # 对存储区内数据降维，只保留信息量最大的数据，去除冗余数据
    x_pca = mem.cpu().T.numpy()
    x_std = scale(x_pca, with_mean=True, with_std=True, axis=0)

    pca = PCA(n_components = 0.99)
    pca.fit(x_std)
    mem = pca.transform(x_std)
    mem = torch.Tensor(mem).T.cuda()

    Arc_p[Arc_p_index[0]][Arc_p_index[1]][Arc_p_index[2]][Arc_p_index[3]] = mem


# def pro_weight(p, Arc_p_index, x, w, filename, stride=1):
def pro_weight(p, Arc_p_index, w, filename, stride=1):
    # # 查看实时x,w情况
    # folder = os.path.exists("绝对地址" + filename)
    # if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
    #     os.makedirs("绝对地址" + filename)  # makedirs 创建文件时如果路径不存在会创建这个路径
    # # with open("绝对地址" + filename + "/x.txt", "a") as f:
    # #     f.write(str(x) + '\n')
    # with open("绝对地址" + filename + "/w.txt", "a") as f:
    #     f.write(str(w) + '\n')

    # w.grad.data更新
    F, _, W, H = w.shape
    if not p.cpu().equal(torch.eye(1)):
        if W != 1 and H != 1:
            t = torch.mm(p, p.T)
            try:
                mem_1 = torch.linalg.inv(t)
            except:
                mem_1 = torch.linalg.pinv(t)
            pro = torch.mm(torch.mm(p.T, mem_1), p)  # 计算投影
            w.grad.data = torch.mm(w.grad.data.view(F, -1), torch.t(pro.data)).view_as(w)  # 按改进方法更新参数


#如果是reduction  cell ,对于一开始的输入 s0,s1到第一个中间节点的连接，stride=2，featuremap实现减半
def train(train_queue, valid_queue, model, architect, arch_archive,n_archive_recent,r_archive_recent, criterion, optimizer, lr, filename, epoch, epochs):
    objs = utils.AvgrageMeter() # 用于保存loss的值
    top1 = utils.AvgrageMeter() # 前1预测正确的概率
    top5 = utils.AvgrageMeter() # 前5预测正确的概率
    num = 0
    # 取一次batch_size操作指一次interation,执行一次for循环体，共需要执行step（总数量/batch_size）次iteration
    # 相当于每一个step取出一个batch

    for step, (input, target) in enumerate(train_queue):
        # model_save=copy.deepcopy(model)
        model.train()
        n = input.size(0)
        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()
        # tensor([8, 2, 4, 1, 4, 2, 5, 8, 2, 3, 7, 2, 9, 4, 9, 2, 7, 3, 3, 7, 7, 5, 7, 1,        9, 3, 1, 7, 6, 3, 7, 8])

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))  #用于架构参数更新的一个batch 。使用iter(dataloader)返回的是一个迭代器，然后可以使用next访问；
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda()
        
        #对α进行更新，对应伪代码的第一步，也就是用公式6
        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
        
        arch_param_save = model.arch_parameters()
    
        temp = opt.initial_temp * np.exp(-opt.anneal_rate * step)
        temperature=torch.tensor([temp]).type(torch.FloatTensor)
        # alpha_nor=torch.tensor(arch_param_save[0]).type(torch.FloatTensor)
        alpha_nor=arch_param_save[0].clone().detach().type(torch.FloatTensor)
        Z1= torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
            temperature, logits = alpha_nor)
        Z_nor=Z1.sample()
        alpha_red=arch_param_save[1].clone().detach().type(torch.FloatTensor)
        # alpha_red=torch.tensor(arch_param_save[1]).type(torch.FloatTensor)
        Z2 = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
            temperature, logits = alpha_red)  
        Z_red = Z2.sample()
        
        gene_normal = _parse_gene(F.softmax(Z_nor, dim=-1).data.cpu().numpy())
        gene_reduce = _parse_gene(F.softmax(Z_red, dim=-1).data.cpu().numpy())
        arch_gene = [gene_normal,gene_reduce]
        
        model_weights=get_weights_from_arch(arch_gene)        
        model = set_model_weights(model, model_weights)
        
        # logits, x_list = model(input)
        logits = model(input)
        loss = criterion(logits, target)
                     
        # 对w进行更新，对应伪代码的第二步
        optimizer.zero_grad()  # 清除之前的更新参数值梯度
        loss.backward()  # 反向传播，计算梯度

        # # 记录梯度
        # folder = os.path.exists("地址"+filename)
        # if not folder:                   # 判断是否存在文件夹如果不存在则创建为文件夹
        #     os.makedirs("地址"+filename)   # makedirs 创建文件时如果路径不存在会创建这个路径
        # for name, param in model.named_parameters():
        #     with open("地址"+filename+"/weights.txt", "a") as f:
        #         f.write(str(name)+' ')
        #         f.write(str(param.size())+'\n')
        
        # with open("地址"+filename+"/weights.txt", "a") as f:
        #     f.write("input"+str(input))
        #     f.write(str(0) + '\n')

        
        # 目前gene_normal[i][2]表示的是输入的节点，将其转化为14条输入边的序号append到gene_normal[i][3]上，为后面和编号为14的weights匹配做准备
        new_gene_normal = gene_normal
        for i in range(0, len(gene_normal)):
            if i in [0, 1]:
                new_gene_normal[i] = list(new_gene_normal[i])
                new_gene_normal[i].append(new_gene_normal[i][1]+0)
                new_gene_normal[i] = tuple(new_gene_normal[i])
            if i in [2, 3]:
                new_gene_normal[i] = list(new_gene_normal[i])
                new_gene_normal[i].append(new_gene_normal[i][1]+2)
                new_gene_normal[i] = tuple(new_gene_normal[i])
            if i in [4, 5]:
                new_gene_normal[i] = list(new_gene_normal[i])
                new_gene_normal[i].append(new_gene_normal[i][1]+5)
                new_gene_normal[i] = tuple(new_gene_normal[i])
            if i in [6, 7]:
                new_gene_normal[i] = list(new_gene_normal[i])
                new_gene_normal[i].append(new_gene_normal[i][1]+9)
                new_gene_normal[i] = tuple(new_gene_normal[i])

        for name, param in model.named_parameters():
            param = param
            # 将权重名称转化为可识别字符的短语
            params_name = name
            # params_name = params_name.decode("utf-8")
            params_name = params_name.split('.')
            # 进行初步筛查
            weights_num = 0  # 记录可操作参数个数
            if len(params_name) in [8, 9]:

                # 看一下是第几个cell
                params_cell_number = params_name[1]

                # 判断是normal cell还是reduce cell
                if params_cell_number in ['0', '1', '3', '4', '6', '7']:  # 判断块是否是normal
                    for i in range(0, len(new_gene_normal)):
                        # params_name[3]: 14条中的哪条
                        if params_name[3] == str(new_gene_normal[i][2]):
                            if new_gene_normal[i][0] in [3, 4, 5, 6, 7]:
                                # params_name[5]: 8个操作中的哪个
                                if params_name[5] == str(new_gene_normal[i][0]):
                                    # 执行文章算法，此时state[gene_normal[i][1]]为x
                                    weights_num = weights_num + 1
                                    # params_name[7]: op中的第几歩操作
                                    params_cell_number = int(params_cell_number)
                                    Arc_p_index = [params_cell_number, int(params_name[3]), int(params_name[5]),int(params_name[7])]
                                    p = Arc_p[Arc_p_index[0]][Arc_p_index[1]][Arc_p_index[2]][Arc_p_index[3]]
                                    # pro_weight(p, Arc_p_index, x, param, filename, conv_class)
                                    pro_weight(p, Arc_p_index, param, filename)

                elif (params_cell_number in ['2', '5']):
                    # 判断块是否是reduce
                    # 每个cell内部，如果ops号等于new_gene_normal[i][3],
                    # 判断new_gene_normal[i][0]是不是属于[4,5,6,7],
                    # 不属于，直接continue；属于，再判断和new_gene_normal[i][0]相等的ops右号，
                    # 留下来，选取数值为new_gene_normal[i][2]的state元素作为x
                    for i in range(0, len(new_gene_normal)):
                        if params_name[3] == str(new_gene_normal[i][2]):
                            if new_gene_normal[i][0] in [3, 4, 5, 6, 7]:
                                if params_name[5] == str(new_gene_normal[i][0]):
                                    # 执行文章算法，此时state[gene_normal[i][1]]为x
                                    weights_num = weights_num + 1

                                    params_cell_number = int(params_cell_number)

                                    if int(params_name[5]) == 3:
                                        conv_num = params_name[6].split('_')
                                        Arc_p_index = [params_cell_number, int(params_name[3]), int(params_name[5]),int(conv_num[1])]
                                    else:
                                        
                                        Arc_p_index = [params_cell_number, int(params_name[3]), int(params_name[5]),int(params_name[7])]

                                    p = Arc_p[Arc_p_index[0]][Arc_p_index[1]][Arc_p_index[2]][Arc_p_index[3]]
                                    pro_weight(p, Arc_p_index, param, filename)

        # Apply step

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度剪切
        # nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()  # 更新网络内部的权重，应用梯度

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)
        
        n_archive_recent.extend([arch_gene[0]])
        r_archive_recent.extend([arch_gene[1]])
        n_archive_recent=n_archive_recent[-50:]   # limitate the number for architecture_recent as 50, a.k.a. C=50 in the paper
        r_archive_recent=r_archive_recent[-50:]
        
        arch_archive_new = arch_archive_update(arch_gene,arch_archive,n_archive_recent,r_archive_recent)
        
        alphas_normal = Variable(Z_nor.cuda(), requires_grad=True)
        alphas_reduce = Variable(Z_red.cuda(), requires_grad=True)
        arch_parameters = [alphas_normal,alphas_reduce]
        
        model=set_model_weights(model, arch_parameters)  # set back

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        num = num + 1

    with open("./model_list.txt","a") as f:
            # f.write(str(model_list))
            for i in model_acc_list:
                f.write(str(i)+" "+str(model_acc_list[i]))
                f.write('\n')
            f.write('\n\n')

    for step, (input, target) in enumerate(train_queue):
        # model_save=copy.deepcopy(model)
        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()

        arch_param_save = model.arch_parameters()

        temp = opt.initial_temp * np.exp(-opt.anneal_rate * step)
        temperature = torch.tensor([temp]).type(torch.FloatTensor)
        # alpha_nor=torch.tensor(arch_param_save[0]).type(torch.FloatTensor)
        alpha_nor = arch_param_save[0].clone().detach().type(torch.FloatTensor)
        Z1 = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
            temperature, logits=alpha_nor)
        Z_nor = Z1.sample()
        gene_normal = _parse_gene(F.softmax(Z_nor, dim=-1).data.cpu().numpy())

        logits = model(input)
        loss = criterion(logits, target)

        optimizer.zero_grad()  # 清除之前的更新参数值梯度
        loss.backward()  # 反向传播，计算梯度

        new_gene_normal = gene_normal
        for i in range(0, len(gene_normal)):
            if i in [0, 1]:
                new_gene_normal[i] = list(new_gene_normal[i])
                new_gene_normal[i].append(new_gene_normal[i][1] + 0)
                new_gene_normal[i] = tuple(new_gene_normal[i])
            if i in [2, 3]:
                new_gene_normal[i] = list(new_gene_normal[i])
                new_gene_normal[i].append(new_gene_normal[i][1] + 2)
                new_gene_normal[i] = tuple(new_gene_normal[i])
            if i in [4, 5]:
                new_gene_normal[i] = list(new_gene_normal[i])
                new_gene_normal[i].append(new_gene_normal[i][1] + 5)
                new_gene_normal[i] = tuple(new_gene_normal[i])
            if i in [6, 7]:
                new_gene_normal[i] = list(new_gene_normal[i])
                new_gene_normal[i].append(new_gene_normal[i][1] + 9)
                new_gene_normal[i] = tuple(new_gene_normal[i])

        for name, param in model.named_parameters():
            param = param
            # 将权重名称转化为可识别字符的短语
            params_name = name
            # params_name = params_name.decode("utf-8")
            params_name = params_name.split('.')
            # 进行初步筛查
            weights_num = 0  # 记录可操作参数个数
            if len(params_name) in [8, 9]:

                params_cell_number = params_name[1]

                if params_cell_number in ['0', '1', '3', '4', '6', '7']:  # 判断块是否是normal
                    for i in range(0, len(new_gene_normal)):
                        if params_name[3] == str(new_gene_normal[i][2]):
                            if new_gene_normal[i][0] in [3, 4, 5, 6, 7]:
                                if params_name[5] == str(new_gene_normal[i][0]):
                                    weights_num = weights_num + 1
                                    params_cell_number = int(params_cell_number)
                                    Arc_p_index = [params_cell_number, int(params_name[3]), int(params_name[5]), int(params_name[7])]
                                    p = Arc_p[Arc_p_index[0]][Arc_p_index[1]][Arc_p_index[2]][Arc_p_index[3]]
                                    arc_update(p, Arc_p_index, param)

                elif (params_cell_number in ['2', '5']):
                    for i in range(0, len(new_gene_normal)):
                        if params_name[3] == str(new_gene_normal[i][2]):
                            if new_gene_normal[i][0] in [3, 4, 5, 6, 7]:
                                if params_name[5] == str(new_gene_normal[i][0]):
                                    weights_num = weights_num + 1
                                    params_cell_number = int(params_cell_number)
                                    if int(params_name[5]) == 3:
                                        conv_num = params_name[6].split('_')
                                        Arc_p_index = [params_cell_number, int(params_name[3]), int(params_name[5]), int(conv_num[1])]
                                    else:
                                        Arc_p_index = [params_cell_number, int(params_name[3]), int(params_name[5]), int(params_name[7])]
                                    p = Arc_p[Arc_p_index[0]][Arc_p_index[1]][Arc_p_index[2]][Arc_p_index[3]]
                                    arc_update(p, Arc_p_index, param)
    
    return top1.avg, objs.avg, n_archive_recent, r_archive_recent

sub_arc_p = []
for i in range(14):
    for j in range(8):
        if j in range(3):
            sub_arc_p.append([torch.eye(1)])
        elif j in [3, 6, 7]:
            sub_arc_p.append([torch.eye(1), torch.eye(1), torch.eye(1), torch.eye(1)])
        else:
            sub_arc_p.append([torch.eye(1), torch.eye(1), torch.eye(1), torch.eye(1), torch.eye(1), torch.eye(1), torch.eye(1), torch.eye(1)])
sub_arc_p = np.array(sub_arc_p, dtype=object).reshape(14, 8)
Arc_p = [sub_arc_p for i in range(8)]    #生成的8个Arc，对应8个cell
Arc_p = np.array(Arc_p)



def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    
    arch_param_save=model.arch_parameters()
    arch_gene=model.gene()##############get the encode of best arch(arch=[[0,1,0,2],[0,1,0,2]])
    model_weights=get_weights_from_arch(arch_gene)        ###########################
    model=set_model_weights(model,model_weights)#############
    
    

    for step, (input, target) in enumerate(valid_queue):
        # with torch.no_grad():
        input = Variable(input).cuda()
        target = Variable(target).cuda()
        # input = Variable(input, volatile=True).cuda()
        # target = Variable(target, volatile=True).cuda(async=True)

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    # model=set_model_weights(model,arch_param_save)

    return top1.avg, objs.avg



if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

np.random.seed(args.seed)
torch.cuda.set_device(args.gpu)
cudnn.benchmark =True
torch.manual_seed(args.seed)
cudnn.enabled=True
torch.cuda.manual_seed(args.seed)
logging.info('gpu device = %d' % args.gpu)
logging.info("args = %s", args)

criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()

model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
model = model.cuda(args.gpu)
logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

optimizer = torch.optim.SGD(
    model.parameters(),
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay)

# train_transform, valid_transform = utils._data_transforms_cifar10(args)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_data = dset.ImageFolder(
    args.data,
    transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2),
      transforms.ToTensor(),
      normalize,
    ]))

valid_data = dset.ImageFolder(
    args.data,
    transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ]))

# train_data = dset.ImageFolder(root=args.data, train=True, 
#     download=True, transform=train_transform)

num_train = len(train_data)
indices = list(range(num_train))
split = int(np.floor(args.train_portion * num_train))
split_end = int(num_train)

train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      shuffle=False,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=4)

valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      shuffle=False,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:split_end]),
      pin_memory=True, num_workers=4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, float(args.epochs), eta_min=args.learning_rate_min)

from architect import Architect

architect = Architect(model, args)

arch_archive=[]
arch_gen1=random_arch_generate()
arch_gen2=random_arch_generate()


n_archive_recent=[arch_gen1]
r_archive_recent=[arch_gen2]

record_train_acc=[]
record_valid_acc=[]
record_valid_acc_retrain=[]

model_list = []
model_acc_list = {}

for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    # training
    train_acc, train_obj,n_archive_recent,r_archive_recent = train(train_queue, valid_queue, model, architect, arch_archive, n_archive_recent,
                                                                   r_archive_recent, criterion, optimizer, lr, args.save, epoch, args.epochs)
    # torch.cuda.empty_cache()
    logging.info('train_acc %f', train_acc)

    # 当前模型加入以往模型集合队列中
    model_list.append(model)
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)
    logging.info('valid_obj %f', valid_obj)

    if epoch % 40 == 0:
        best_top1 = 0
        best_model = model
        for model in model_list:
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            if valid_acc > best_top1:
                best_top1 = valid_acc
                best_model = model
        logging.info('best_top1 %f', best_top1)
        logging.info('best_model %f', model.genotype())

    utils.save(model, os.path.join(args.save, 'weights.pt'))
    


