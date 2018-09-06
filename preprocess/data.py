import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt



def test_sythesis(test_data_dir,output_dir,label='ZJL211'):
    
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

if __name__ == '__main__':
    change_space_to_tab('../../../../Downloads/submit.txt')

    # data_dir = '../source/DatasetA_20180813'
    # output_dir = '../source/DatasetA_20180813/processed/group'
    # get_data_statics(data_dir,output_dir)
    # test_data_dir = '../source/DatasetA_20180813/test'
    # test_sythesis(test_data_dir,output_dir)
    # group_class_label(data_dir,output_dir)
