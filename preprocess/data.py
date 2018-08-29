import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt



def get_data_statics(data_dir,output_dir,val_label_num = 20,feature_val_ratio = 0.1):

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

    data_dir = '../source/DatasetA_20180813'
    output_dir = '../source/DatasetA_20180813/processed'
    get_data_statics(data_dir,output_dir)