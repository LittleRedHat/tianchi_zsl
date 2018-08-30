python feat_train.py \
        --epoches 50 \
        --train_file ../source/DatasetA_20180813/processed/feature_train.txt \
        --val_file ../source/DatasetA_20180813/processed/feature_val.txt \
        --image_dir ../source/DatasetA_20180813/train \
        --embed_file ../source/DatasetA_20180813/class_wordembeddings.txt \
        --lable_name_file ../source/DatasetA_20180813/processed/train_label.txt \
        --attribute_file ../source/DatasetA_20180813/attributes_per_class.txt \
        --backbone mobilenet \
        --lr 0.001 \
        --optim adam \
        --nw 8 \
        --batch_size 192 \
        --device_ids 4,5,6,7


