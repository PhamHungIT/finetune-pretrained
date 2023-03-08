# Note: Comment lines of part that don't execute

# Demo small

# python train.py \
#     --train_file ../data/small/small_train.csv \
#     --val_file ../data/small/small_test.csv \
#     --text_col sentences \
#     --label_col category \
#     --vectorizer bow \
#     --checkpoint_dir ../checkpoints/small_demo


# Train

# python train.py \
#     --train_file ../data/cosmetic/l2/train_l2.csv \
#     --val_file ../data/cosmetic/l2/val_l2.csv \
#     --text_col product_name \
#     --label_col category \
#     --vectorizer bow \
#     --checkpoint_dir ../checkpoints/cosmetic_l2/bow-mlp

# python train.py \
#     --train_file ../data/cosmetic/l2/train_l2.csv \
#     --val_file ../data/cosmetic/l2/val_l2.csv \
#     --text_col product_name \
#     --label_col category \
#     --vectorizer tf_idf \
#     --checkpoint_dir ../checkpoints/cosmetic_l2/tf_idf-mlp



# Evaluate - Infer

# python infer.py \
#     --test_path ../data/cosmetic/test_ctv.csv \
#     --vectorizer word2vec \
#     --checkpoint_dir ../checkpoints/cosmetic_l1/w2v-mlp \
#     --text_col product_name \
#     --category_col label_l1
    

# python infer.py \
#     --test_path ../data/cosmetic/l2/test_cosmetics_V1_20230603.csv\
#     --vectorizer bow \
#     --checkpoint_dir ../checkpoints/cosmetic_l2/bow-mlp \
#     --text_col product_name \
#     --category_col label_l2
    
python infer.py \
    --test_path ../data/cosmetic/l2/test_cosmetics_V1_20230603.csv\
    --vectorizer tf_idf \
    --checkpoint_dir ../checkpoints/cosmetic_l2/tf_idf-mlp \
    --text_col product_name \
    --category_col label_l2
    


# python evaluate.py \
#     --test_path "../data/raw/sample_test.csv" \
#     --checkpoint_path "../checkpoints/demo/last.pt"
    