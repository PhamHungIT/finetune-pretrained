# Note: Comment lines of part that don't execute

# Build data
# python build_data.py \
#     --data_dir ../data/_50k \
#     --data_name df_train.csv \
#     --col_text sentences \
#     --col_category category \
#     --split_type train
    # --split_file

python train.py \
    --train_file ../data/_50k/df_train.csv \
    --val_file ../data/_50k/df_val.csv \
    --text_col sentences \
    --label_col category \
    --vectorizer tf_idf \
    --checkpoint_dir ../checkpoints


# python run.py \
#     --train_path "../data/raw/product.csv"

# python evaluate.py \
#     --test_path "../data/raw/sample_test.csv" \
#     --checkpoint_path "../checkpoints/demo/last.pt"
    