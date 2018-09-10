import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from model.features import DenseNet
import torch
import random
import json


def hierarchy_label(data_dir):
    label_to_attribute = {}
    group_num = 6

    with open(os.path.join(data_dir,'attributes_per_class.txt')) as f:
        for line in f:
            label_attribute = line.strip().split()
            label,attribute = label_attribute[0],label_attribute[1:]
            if label in label_to_name:
                attribute = [float(i) for i in attribute]
                label_to_attribute[label] = attribute

    groups = [[] for _ in range(group_num)]
    label_to_group = {}
    for label,attribute in label_to_attribute.items():
        for i in range(group_num):
            if attribute[i] == 1.0:
                groups[i].append(label)
                label_to_group[label] = i
                break

    
    
    with open(os.path.join(data_dir,'processed','feature_train.txt')) as f:
        for line in f:
            image_name,label = line.strip().split()

            group = label_to_group[label]
            group_index = groups

def test_sythesis(test_data_dir,output_dir,label='ZJL22'):
    
    imgs = os.listdir(test_data_dir)
    with open(os.path.join(output_dir,'test.txt'),'w') as f:
        for img_name in imgs:
            if '.jpg' in img_name:
                f.write('{}\t{}\n'.format(img_name,label))

def group_class_label(data_dir,output_dir):
    group_names = ['animal','transportation','clothes','plant','tableware','device']
    groups = [[],[],[],[],[],[]]

    label_to_name = {}
    with open(os.path.join(data_dir,'processed/trainval_label.txt')) as f:
        for line in f:
            label,name = line.strip().split()
            label_to_name[label] = name
    label_to_attribute = {}
    with open(os.path.join(data_dir,'attributes_per_class.txt')) as f:
        for line in f:
            label_attribute = line.strip().split()
            label,attribute = label_attribute[0],label_attribute[1:]
            if label in label_to_name:
                attribute = [float(i) for i in attribute]
                label_to_attribute[label] = attribute

    for label,attribute in label_to_attribute.items():
        group_attribute = attribute[:len(groups)]
        for i,attr in enumerate(group_attribute):
            if attr == 1:
                groups[i].append(label)
    for i,name in enumerate(group_names):
        with open(os.path.join(output_dir,'{}_label.txt'.format(name)),'w') as f:
            for label in groups[i]:
                f.write('{}\t{}\n'.format(label,label_to_name[label]))
    label_to_imgname = {}
    with open(os.path.join(data_dir,'train.txt')) as f:
        for line in f:
            name,label = line.strip().split()
            g = label_to_imgname.get(label,[])
            g.append(name)
            label_to_imgname[label] = g
    for i,name in enumerate(group_names):
        with open(os.path.join(output_dir,'{}.txt'.format(name)),'w') as f:
            for label in groups[i]:
                for img_name in label_to_imgname[label]:
                    f.write('{}\t{}\n'.format(img_name,label))

def change_space_to_tab(input_file):
    name_to_label = {}
    with open(input_file,'r') as f:
        for line in f:
            name,label = line.strip().split()
            name_to_label[name] = label
    with open('submit.txt','w') as f:
        for name,label in name_to_label.items():
            f.write('{}\t{}\n'.format(name,label))

def extract_classifier():
    
    model = DenseNet(num_classes=150)
    model.load_state_dict(torch.load('../output/models/densenet/model_best.pth.tar')['model'])
    classifier = model.model.classifier

    
    torch.save(classifier.state_dict(),'classifier.pth.tar')

def classify_image_by_dir(data_dir):
    with open(os.path.join(data_dir,'train.txt')) as f:
        for line in f:
            name, label = line.strip().split()
            src = os.path.join(data_dir,'train',name)
            dir = os.path.join(data_dir,'groups',label)
            if not os.path.isdir(dir):
                os.makedirs(dir)
            dst = os.path.join(dir,name)
            
            os.symlink(src,dst)

def get_data_statics(data_dir,output_dir,val_label_num = 40,feature_val_ratio = 0.15):

    trainval_file_path = 'train.txt'
    label_list_file_path = 'label_list.txt'


    label_to_name = {}

    with open(os.path.join(data_dir,label_list_file_path),'r') as f:
        for line in f:
            label,name = line.strip().split()
            label_to_name[label] = name


    labels = list(label_to_name.keys())


    trainval_df = pd.read_csv(os.path.join(data_dir,trainval_file_path),sep='\t')
    trainval_df.columns = ['image_name','label']

    label_distribution = trainval_df.label.value_counts()

    trainval_labels = list(label_distribution.index)
    trainval_counts = list(label_distribution.values)

    with open(os.path.join(output_dir,'trainval_label.txt'),'w') as f:
        for label in trainval_labels:
            f.write(label+'\t'+label_to_name[label]+'\n')
        
    with open(os.path.join(output_dir,'label_distribution.txt'),'w') as f:
        for index in range(len(trainval_labels)):
            label = trainval_labels[index]
            count = trainval_counts[index]
            f.write('{}\t{}\n'.format(label,count))

    test_labels = [i for i in labels if i not in trainval_labels]

    with open(os.path.join(output_dir,'test_label.txt'),'w') as f:
        for label in test_labels:
            f.write(label+'\t'+label_to_name[label]+'\n')

    ## split train/val by class

    # np.random.shuffle(train_labels)
    val_labels = trainval_labels[:val_label_num]
    train_labels = trainval_labels[val_label_num:]

    with open(os.path.join(output_dir,'val_label.txt'),'w') as f:
        for label in val_labels:
            f.write(label+'\t'+label_to_name[label]+'\n')

    with open(os.path.join(output_dir,'train_label.txt'),'w') as f:
        for label in train_labels:
            f.write(label+'\t'+label_to_name[label]+'\n')
    

    train_samples = trainval_df[trainval_df['label'].isin(train_labels)]
    val_samples = trainval_df[trainval_df['label'].isin(val_labels)]

    train_samples.to_csv(os.path.join(output_dir,'train.txt'),sep='\t',header=False,index=False)
    val_samples.to_csv(os.path.join(output_dir,'val.txt'),sep='\t',header=False,index=False)

    ## split feature extractor samples 10%
    feature_trainval_samples = train_samples.to_dict(orient='records')
    np.random.shuffle(feature_trainval_samples)
    

    feature_val_sample_num = int(len(feature_trainval_samples) * feature_val_ratio)


    feature_val_samples = feature_trainval_samples[:feature_val_sample_num]
    feature_train_samples = feature_trainval_samples[feature_val_sample_num:]

    with open(os.path.join(output_dir,'feature_train.txt'),'w') as f:
        for sample in feature_train_samples:
            f.write('{}\t{}\n'.format(sample['image_name'],sample['label']))

    with open(os.path.join(output_dir,'feature_val.txt'),'w') as f:
        for sample in feature_val_samples:
            f.write('{}\t{}\n'.format(sample['image_name'],sample['label']))


def attribute_distribution(data_dir,output_dir,val_label_num = 20,val_sample_ratio = 0.1):
    label_df = pd.read_csv(os.path.join(data_dir,'processed','class_to_attribute.csv'))
    ## has no group labels
    animal_df = label_df[label_df['is animal'] == 1]
    transportation_df = label_df[label_df['is transportation'] == 1]
    clothes_df = label_df[label_df['is clothes'] == 1]
    plant_df = label_df[label_df['is plant'] == 1]
    tableware_df = label_df[label_df['is tableware'] == 1]
    device_df = label_df[label_df['is device'] == 1]
    
    animal_in_train = animal_df[animal_df['is in train'] == 1]
    transportation_in_train = transportation_df[transportation_df['is in train'] == 1]
    clothes_in_train = clothes_df[clothes_df['is in train'] == 1]
    plant_in_train = plant_df[plant_df['is in train'] == 1]
    tableware_in_train = tableware_df[tableware_df['is in train'] == 1]
    device_in_train = device_df[device_df['is in train'] == 1]

    info = 'animal:{}/{} transportation:{}/{} clothes:{}/{} plant:{}/{} tableware:{}/{} device:{}/{}'.format(\
        len(animal_in_train),len(animal_df),\
        len(transportation_in_train),len(transportation_df),\
        len(clothes_in_train),len(clothes_df),\
        len(plant_in_train),len(plant_df),\
        len(tableware_in_train),len(tableware_df),\
        len(device_in_train),len(device_df))

    print(info)
    

    samples = pd.read_csv(os.path.join(data_dir,'train.txt'),sep='\t')
    samples.columns = ['image_name','label']

    animal_samples = samples[samples['label'].isin(animal_df['class'].values)]
    
    device_samples = samples[samples['label'].isin(device_df['class'].values)]

    transportation_samples = samples[samples['label'].isin(transportation_df['class'].values)]

    plant_samples = samples[samples['label'].isin(plant_df['class'].values)]

    animal_samples.to_csv(os.path.join(output_dir,'animal.txt'),sep='\t',index=False,header=False)
    animal_df[['class','class name']].to_csv(os.path.join(output_dir,'animal_label.txt'),sep='\t',index=False,header=False)
    animal_in_train[['class','class name']].to_csv(os.path.join(output_dir,'animal_trainval_label.txt'),sep='\t',index=False,header=False)

    transportation_samples.to_csv(os.path.join(output_dir,'trans.txt'),sep='\t',index=False,header=False)
    transportation_df[['class','class name']].to_csv(os.path.join(output_dir,'trans_label.txt'),sep='\t',index=False,header=False)
    transportation_in_train[['class','class name']].to_csv(os.path.join(output_dir,'trans_trainval_label.txt'),sep='\t',index=False,header=False)


    device_samples.to_csv(os.path.join(output_dir,'device.txt'),sep='\t',index=False,header=False)
    device_df[['class','class name']].to_csv(os.path.join(output_dir,'device_label.txt'),sep='\t',index=False,header=False)
    device_in_train[['class','class name']].to_csv(os.path.join(output_dir,'device_trainval_label.txt'),sep='\t',header=False)

    plant_samples.to_csv(os.path.join(output_dir,'plant.txt'),sep='\t',index=False,header=False)
    plant_df[['class','class name']].to_csv(os.path.join(output_dir,'plant_label.txt'),sep='\t',index=False,header=False)
    plant_in_train[['class','class name']].to_csv(os.path.join(output_dir,'plant_trainval_label.txt'),sep='\t',header=False)

    ## split animal trans plant device
    ratios = [16,5,9,10]
    nums = [ i * val_label_num  // sum(ratios) for i in ratios]

    animal_label = list(animal_in_train['class'])
    random.shuffle(animal_label)
    animal_val_label = animal_label[:nums[0]]
    animal_train_label = animal_label[nums[0]:]

    trans_label = list(transportation_in_train['class'])
    random.shuffle(trans_label)
    trans_val_label = trans_label[:nums[1]]
    trans_train_label = trans_label[nums[1]:]

    plant_label = list(plant_in_train['class'])
    random.shuffle(plant_label)
    plant_val_label = plant_label[:nums[1]]
    plant_train_label = plant_label[nums[1]:]

    device_label = list(device_in_train['class'])
    random.shuffle(device_label)
    device_val_label = device_label[:nums[1]]
    device_train_label = device_label[nums[1]:]
    
    

    all_val_label = animal_val_label + trans_val_label + plant_val_label + device_val_label

    all_train_label = animal_train_label + trans_train_label + plant_train_label + device_train_label 

    val_label_df = label_df[label_df['class'].isin(all_val_label)]

    train_label_df = label_df[label_df['class'].isin(all_train_label)]

    val_samples = samples[samples['label'].isin(all_val_label)]
    train_samples = samples[samples['label'].isin(all_train_label)]


    val_samples.to_csv(os.path.join(output_dir,'val.txt'),index=False,header=False,sep='\t')

    train_samples = train_samples.sample(frac=1).reset_index(drop=True)

    train_samples.to_csv(os.path.join(output_dir,'train.txt'),index=False,header=False,sep='\t')

    feature_val = int(len(train_samples) * val_sample_ratio)
    feature_train_samples = train_samples[feature_val:]
    feature_val_samples = train_samples[:feature_val]

    feature_train_samples.to_csv(os.path.join(output_dir,'feature_train.txt'),index=False,header=False,sep='\t')
    feature_val_samples.to_csv(os.path.join(output_dir,'feature_val.txt'),index=False,header=False,sep='\t')

    groups = {
        'groups':[animal_train_label,trans_train_label,plant_train_label,device_train_label],
        'names':['animal','trans','plant','device']   
    }
    with open(os.path.join(output_dir,'groups.json'),'w') as f:
        json.dump(groups,f)

    
    




    
    # useful_labels = label_df[label_df['is animal'] == ]
    


if __name__ == '__main__':
    # change_space_to_tab('../../../../Downloads/submit.txt')

    data_dir = '../source/DatasetA_20180813'
    output_dir = '../source/DatasetA_20180813/processed/group'
    # get_data_statics(data_dir,output_dir)
    # test_data_dir = '../source/DatasetA_20180813/test'
    # output_dir = '../source/DatasetA_20180813/processed'
    # test_sythesis(test_data_dir,output_dir)
    # group_class_label(data_dir,output_dir)
    # extract_classifier()
    # classify_image_by_dir('/Users/zookeeper/Documents/workspace/ali-zsl/source/DatasetA_20180813')
    attribute_distribution(data_dir,output_dir)


