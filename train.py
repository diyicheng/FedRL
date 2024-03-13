# We are committed to fully open-source the paper upon acceptance to promote the open-source community.
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from models import FedRLRecommender
import numpy as np
import argparse
import time
import pickle
import os
import matplotlib.pyplot as plt
import json
import io
import tensorflow as tf


def ndcg(true_y, pred_y, query_size=5):
    sorted_truth = tf.math.top_k(true_y, k=query_size)
    sorted_truth_idx = sorted_truth.indices.numpy()
    sorted_truth_score = sorted_truth.values.numpy()
    idcg = 0.0
    for idx, s in enumerate(sorted_truth_score):
        idcg = idcg + 1.0 / np.log2(idx + 2)  # start from 0.
    dcg_5 = 0.0
    dcg_10 = 0.0
    dcg_20 = 0.0
    dcg_50 = 0.0
    sorted_pred = tf.math.top_k(pred_y, k=50)
    sorted_pred_score = sorted_pred.values.numpy()
    sorted_pred_idx = sorted_pred.indices.numpy()

    if sorted_pred_score[0] == sorted_pred_score[49]:
        return 0, 0, 0, 0
    for idx, s in enumerate(sorted_pred_score):
        if sorted_pred_idx[idx] in sorted_truth_idx:
            if idx < 5:
                dcg_5 = dcg_5 + 1.0 / np.log2(idx + 2)
            if idx < 10:
                dcg_10 = dcg_10 + 1.0 / np.log2(idx + 2)
            if idx < 20:
                dcg_20 = dcg_20 + 1.0 / np.log2(idx + 2)
            if idx < 50:
                dcg_50 = dcg_50 + 1.0 / np.log2(idx + 2)
    return dcg_5 / idcg, dcg_10 / idcg, dcg_20 / idcg, dcg_50 / idcg

def hits(true_y, pred_y, query_size=5):
    sorted_truth = tf.math.top_k(true_y, k=query_size)
    sorted_truth_idx = sorted_truth.indices.numpy()
    sorted_truth_score = sorted_truth.values.numpy()
    sorted_pred = tf.math.top_k(pred_y, k=50)
    sorted_pred_score = sorted_pred.values.numpy()
    sorted_pred_idx = sorted_pred.indices.numpy()
    hits_5 = 0
    hits_10 = 0
    hits_20 = 0
    hits_50 = 0
    if sorted_pred_score[0] == sorted_pred_score[49]:
        return 0, 0, 0, 0
    for idx, s in enumerate(sorted_pred_score):
        if sorted_pred_idx[idx] in sorted_truth_idx:  # actually it's 0-5
            if idx < 5:
                hits_5 += 1
            if idx < 10:
                hits_10 += 1
            if idx < 20:
                hits_20 += 1
            if idx < 50:
                hits_50 += 1
    return hits_5, hits_10, hits_20, hits_50

def get_dataset(user_ids, enriched_ratings, all_item_id, all_item_feature,
                max_genre_size, max_creator_size, max_character_size,
                train_item_id, NEGATIVE_EXAMPLE_NUM, SUPPORT_SIZE, QUERY_SIZE):
    # for k,v in enriched_ratings.items():
    for u_id in user_ids:
        v = enriched_ratings[str(u_id)]
        if len(v) >= (SUPPORT_SIZE + QUERY_SIZE):
            user_picked_item_id = set()
            np_neg_user_train_data = None
            for rec in v:
                if use_existing_items:
                    if int(rec[1]) in train_item_id:
                        user_picked_item_id.add(int(rec[1]))
                else:
                    if int(rec[1]) not in train_item_id:
                        user_picked_item_id.add(int(rec[1]))
            sampled_data_ind = np.random.choice(list(range(len(v))), size=SUPPORT_SIZE + QUERY_SIZE, replace=False)
            np_user_train_data = None
            for idx in sampled_data_ind:
                rec = v[idx]
                user_id = np.asarray(int(rec[0]), dtype=np.int32).reshape(1, -1)
                item_id = np.asarray(int(rec[1]), dtype=np.int32).reshape(1, -1)
                score = np.asarray(float(rec[2]), dtype=np.float32).reshape(1, -1)
                score = np.ones_like(score)
                timestamp = np.asarray(int(rec[3]), dtype=np.int32).reshape(1, -1)
                genre_feature = np.ones(max_genre_size, dtype=np.int32) * 1000000
                creator_feature = np.ones(max_creator_size, dtype=np.int32) * 1000000
                character_feature = np.ones(max_character_size, dtype=np.int32) * 1000000
                l = len(rec[4])
                genre_feature[:l] = rec[4][:]
                genre_feature = genre_feature.reshape(1, -1)
                l = len(rec[5])
                creator_feature[:l] = rec[5][:]
                creator_feature = creator_feature.reshape(1, -1)
                l = len(rec[6])
                character_feature[:l] = rec[6][:]
                character_feature = character_feature.reshape(1, -1)
                np_record = np.concatenate(
                    (user_id, item_id, score, timestamp,
                     genre_feature, character_feature, creator_feature),
                    axis=1
                )
                if np_user_train_data is not None:
                    np_user_train_data = np.concatenate(
                        (np_user_train_data, np_record),
                        axis=0)
                else:
                    np_user_train_data = np_record
            np_user_train_data = np_user_train_data[np_user_train_data[:, 3].argsort()]
            np_user_train_data = np.delete(np_user_train_data, 3, axis=1)
            negative_item_id = all_item_id - user_picked_item_id
            chosen_negative_item_id = np.random.choice(list(negative_item_id), size=NEGATIVE_EXAMPLE_NUM, replace=False)
            for idx in chosen_negative_item_id:
                n_user_id = user_id
                n_item_id = np.asarray(idx, dtype=np.int32).reshape(1, -1)
                n_score = np.asarray(0.0, dtype=np.float32).reshape(1, -1)
                if np_neg_user_train_data is None:
                    np_neg_user_train_data = np.concatenate((n_user_id, n_item_id, n_score, all_item_feature[idx]),
                                                            axis=1)
                else:
                    new_neg_record = np.concatenate((n_user_id, n_item_id, n_score, all_item_feature[idx]),
                                                    axis=1)
                    np_neg_user_train_data = np.concatenate((np_neg_user_train_data, new_neg_record), axis=0)
            yield np_user_train_data, np_neg_user_train_data
def test_FedRL(model, optimizer, use_dataset, dataset_user_cnt,
                 SUPPORT_SIZE, QUERY_SIZE, prefix, tf_writer, step):
    acc_ndcg5 = 0.0
    acc_ndcg10 = 0.0
    acc_ndcg20 = 0.0
    acc_ndcg50 = 0.0
    acc_mae = 0.0
    acc_hits5 = 0.0
    acc_hits10 = 0.0
    acc_hits20 = 0.0
    acc_hits50 = 0.0
    variable_state = list()
    for w in model.trainable_weights:
        variable_state.append(w.numpy())
    for idx, (pos_sample, neg_sample) in enumerate(use_dataset):
        start_time = time.time()
        user_id = pos_sample[: SUPPORT_SIZE, 0]
        n_user_id = neg_sample[: SUPPORT_SIZE, 0]
        true_y = pos_sample[:SUPPORT_SIZE, 2]
        n_true_y = neg_sample[:, 2]
        start_idx = 3
        genre_idx = pos_sample[:SUPPORT_SIZE, start_idx: start_idx + max_genre_size]
        n_genre_idx = neg_sample[:SUPPORT_SIZE, start_idx: start_idx + max_genre_size]
        start_idx = start_idx + max_genre_size
        character_idx = pos_sample[:SUPPORT_SIZE, start_idx: start_idx + max_character_size]
        n_character_idx = neg_sample[:SUPPORT_SIZE, start_idx: start_idx + max_character_size]
        start_idx = start_idx + max_character_size
        creator_idx = pos_sample[:SUPPORT_SIZE, start_idx: start_idx + max_creator_size]
        n_creator_idx = neg_sample[:SUPPORT_SIZE, start_idx: start_idx + max_creator_size]
        meta_user_id = tf.reshape(pos_sample[0, 0], [1, -1])
        with tf.GradientTape(persistent=True) as tape:
            model.init_meta_embeddings(meta_user_id)
            pred_y, _ = model(user_id, genre_idx, creator_idx, character_idx)
            loss_value = model.loss(true_y, pred_y)
            grad = tape.gradient(loss_value, [
                model.genre_converter_kernel,
                model.genre_converter_bias,
                model.creator_converter_kernel,
                model.creator_converter_bias,
                model.character_converter_kernel,
                model.character_converter_bias,
                model.item_embedding_converter,
                model.predictor_layer1_kernel,
                model.predictor_layer1_bias,
                model.predictor_layer2_kernel,
                model.predictor_layer2_bias,
                model.predictor_layer3_kernel,
                model.predictor_layer3_bias,
                model.predictor_layer4_kernel
            ])
            user_embed_grad = tape.gradient(loss_value,
                                            model.user_embedding.trainable_weights)
            item_embed_grad = tape.gradient(loss_value,
                                            model.item_embedding.trainable_weights)
            user_preference_idx_grad = tape.gradient(loss_value,
                                                     model.hypernetwork.meta_user_embedding.meta_user_basis.trainable_weights)
        model.local_update(grad, learning_rate)
        optimizer.apply_gradients(zip(item_embed_grad,
                                      model.item_embedding.trainable_weights))
        optimizer.apply_gradients(zip(user_embed_grad,
                                      model.user_embedding.trainable_weights))
        optimizer.apply_gradients(zip(user_preference_idx_grad,
                                      model.hypernetwork.meta_user_embedding.meta_user_basis.trainable_weights))
        with tf.GradientTape(persistent=True) as tape:
            model.init_meta_embeddings(meta_user_id)
            n_pred_y, _ = model(n_user_id, n_genre_idx, n_creator_idx, n_character_idx)
            n_loss_value = model.loss(n_true_y, n_pred_y)
            grad = tape.gradient(n_loss_value, [
                model.genre_converter_kernel,
                model.genre_converter_bias,
                model.creator_converter_kernel,
                model.creator_converter_bias,
                model.character_converter_kernel,
                model.character_converter_bias,
                model.item_embedding_converter,
                model.predictor_layer1_kernel,
                model.predictor_layer1_bias,
                model.predictor_layer2_kernel,
                model.predictor_layer2_bias,
                model.predictor_layer3_kernel,
                model.predictor_layer3_bias,
                model.predictor_layer4_kernel
            ])
            user_embed_grad = tape.gradient(n_loss_value,
                                            model.user_embedding.trainable_weights)
            item_embed_grad = tape.gradient(n_loss_value,
                                            model.item_embedding.trainable_weights)
            user_preference_idx_grad = tape.gradient(n_loss_value,
                                                     model.hypernetwork.meta_user_embedding.meta_user_basis.trainable_weights)

        model.local_update(grad, learning_rate)
        optimizer.apply_gradients(zip(user_embed_grad,
                                      model.user_embedding.trainable_weights))
        optimizer.apply_gradients(zip(item_embed_grad,
                                      model.item_embedding.trainable_weights))
        optimizer.apply_gradients(zip(user_preference_idx_grad,
                                      model.hypernetwork.meta_user_embedding.meta_user_basis.trainable_weights))
        user_id = pos_sample[SUPPORT_SIZE:, 0]
        true_y = pos_sample[SUPPORT_SIZE:, 2].numpy()
        start_idx = 3
        genre_idx = pos_sample[SUPPORT_SIZE:, start_idx: start_idx + max_genre_size]
        start_idx = start_idx + max_genre_size
        character_idx = pos_sample[SUPPORT_SIZE:, start_idx: start_idx + max_character_size]
        start_idx = start_idx + max_character_size
        creator_idx = pos_sample[SUPPORT_SIZE:, start_idx: start_idx + max_creator_size]
        pred_y, att_s = model(user_id, genre_idx, creator_idx, character_idx)
        pred_y = pred_y.numpy()
        batch_mae = mae(true_y, pred_y)
        start_idx = 3
        n_genre_idx = all_item_mat[:, start_idx - 3: start_idx - 3 + max_genre_size]
        start_idx = start_idx + max_genre_size
        n_character_idx = all_item_mat[:, start_idx - 3: start_idx - 3 + max_character_size]
        start_idx = start_idx + max_character_size
        n_creator_idx = all_item_mat[:, start_idx - 3: start_idx - 3 + max_creator_size]
        n_user_id = tf.repeat(user_id[0], repeats=n_genre_idx.shape[0], axis=0)
        pred_score, _ = model(n_user_id, n_genre_idx, n_creator_idx, n_character_idx)
        pred_score = pred_score.numpy()
        pred_y = np.concatenate((pred_y, pred_score), axis=0)
        pred_y = np.squeeze(pred_y)
        ndcg_5, ndcg_10, ndcg_20, ndcg_50 = ndcg(true_y, pred_y, QUERY_SIZE)
        hits_5, hits_10, hits_20, hits_50 = hits(true_y, pred_y, QUERY_SIZE)
        if pred_y[0] != 0:
            acc_ndcg5 += ndcg_5
            acc_ndcg10 += ndcg_10
            acc_ndcg20 += ndcg_20
            acc_ndcg50 += ndcg_50
            acc_mae += batch_mae
            acc_hits5 += hits_5
            acc_hits10 += hits_10
            acc_hits20 += hits_20
            acc_hits50 += hits_50
        exec_time = time.time() - start_time
        tf.print(f"{idx}/{dataset_user_cnt}. \n"
                 f" NDCG@5: {acc_ndcg5 / dataset_user_cnt}. NDCG@10: {acc_ndcg10 / dataset_user_cnt}.\n"
                 f"NDCG@20: {acc_ndcg20 / dataset_user_cnt}. NDCG@50: {acc_ndcg50 / dataset_user_cnt}. \n"
                 f" Hits@5 {acc_hits5 / dataset_user_cnt / QUERY_SIZE}. Hits@10 {acc_hits10 / dataset_user_cnt / QUERY_SIZE}. \n"
                 f"Hits@20 {acc_hits20 / dataset_user_cnt / QUERY_SIZE}. Hits@50 {acc_hits50 / dataset_user_cnt / QUERY_SIZE}.\n"
                 f"{exec_time: .4f}s", end='\r')
    with tf_writer.as_default():
        tf.summary.scalar(prefix + "_NDCG@5", acc_ndcg5 / dataset_user_cnt, step=step)
        tf.summary.scalar(prefix + "_NDCG@10", acc_ndcg10 / dataset_user_cnt, step=step)
        tf.summary.scalar(prefix + "_NDCG@20", acc_ndcg20 / dataset_user_cnt, step=step)
        tf.summary.scalar(prefix + "_NDCG@50", acc_ndcg50 / dataset_user_cnt, step=step)
        tf.summary.scalar(prefix + "_HITS@5", acc_hits5 / dataset_user_cnt / QUERY_SIZE, step=step)
        tf.summary.scalar(prefix + "_HITS@10", acc_hits10 / dataset_user_cnt / QUERY_SIZE, step=step)
        tf.summary.scalar(prefix + "_HITS@20", acc_hits20 / dataset_user_cnt / QUERY_SIZE, step=step)
        tf.summary.scalar(prefix + "_HITS@50", acc_hits50 / dataset_user_cnt / QUERY_SIZE, step=step)
    for idx, w in enumerate(model.trainable_weights):
        model.trainable_weights[idx].assign(variable_state[idx])
def train_FedRL(model, optimizer, dataset_name, basis_optimizer, train_dataset, valid_dataset, test_dataset,
                  learning_rate,
                  train_user_cnt, valid_user_cnt, test_user_cnt, SUPPORT_SIZE, QUERY_SIZE,
                  EPOCH_NUM, tf_writer):
    for epoch_num in range(EPOCH_NUM):
        for idx, (pos_sample, neg_sample) in tqdm(enumerate(train_dataset)):
            time_start = time.time()
            user_id = pos_sample[: SUPPORT_SIZE, 0]
            n_user_id = neg_sample[:SUPPORT_SIZE, 0]
            true_y = pos_sample[:SUPPORT_SIZE, 2]
            n_true_y = neg_sample[:SUPPORT_SIZE, 2]
            start_idx = 3
            genre_idx = pos_sample[:SUPPORT_SIZE, start_idx: start_idx + max_genre_size]
            n_genre_idx = neg_sample[:SUPPORT_SIZE, start_idx: start_idx + max_genre_size]
            start_idx = start_idx + max_genre_size
            character_idx = pos_sample[:SUPPORT_SIZE, start_idx: start_idx + max_character_size]
            n_character_idx = neg_sample[:SUPPORT_SIZE, start_idx: start_idx + max_character_size]
            start_idx = start_idx + max_character_size
            creator_idx = pos_sample[:SUPPORT_SIZE, start_idx: start_idx + max_creator_size]
            n_creator_idx = neg_sample[:SUPPORT_SIZE, start_idx: start_idx + max_creator_size]
            meta_user_id = tf.reshape(pos_sample[0, 0], [1, -1])
            with tf.GradientTape(persistent=True) as tape:
                model.init_meta_embeddings(meta_user_id)
                pred_y, _ = model(user_id, genre_idx, creator_idx, character_idx)
                loss_value = model.loss(true_y, pred_y)
                grad = tape.gradient(loss_value, [
                    model.genre_converter_kernel,
                    model.genre_converter_bias,
                    model.creator_converter_kernel,
                    model.creator_converter_bias,
                    model.character_converter_kernel,
                    model.character_converter_bias,
                    model.item_embedding_converter,
                    model.predictor_layer1_kernel,
                    model.predictor_layer1_bias,
                    model.predictor_layer2_kernel,
                    model.predictor_layer2_bias,
                    model.predictor_layer3_kernel,
                    model.predictor_layer3_bias,
                    model.predictor_layer4_kernel
                ])
                user_embed_grad = tape.gradient(loss_value,
                                                model.user_embedding.trainable_weights)
                item_embed_grad = tape.gradient(loss_value,
                                                model.item_embedding.trainable_weights)
                user_preference_idx_grad = tape.gradient(loss_value,
                                                         model.hypernetwork.meta_user_embedding.meta_user_basis.trainable_weights)
            grad = [tf.clip_by_value(g, -1.0, 1.0) for g in grad]
            item_embed_grad = [tf.clip_by_value(g, -1.0, 1.0) for g in item_embed_grad]
            user_preference_idx_grad = [tf.clip_by_value(g, -1.0, 1.0) for g in user_preference_idx_grad]
            user_embed_grad = [tf.clip_by_value(g, -1.0, 1.0) for g in user_embed_grad]
            model.local_update(grad, learning_rate)
            optimizer.apply_gradients(zip(user_embed_grad,
                                          model.user_embedding.trainable_weights))
            optimizer.apply_gradients(zip(item_embed_grad,
                                          model.item_embedding.trainable_weights))
            optimizer.apply_gradients(zip(user_preference_idx_grad,
                                          model.hypernetwork.meta_user_embedding.meta_user_basis.trainable_weights))
            with tf.GradientTape(persistent=True) as tape:
                model.init_meta_embeddings(meta_user_id)
                n_pred_y, _ = model(n_user_id, n_genre_idx, n_creator_idx, n_character_idx)
                n_loss_value = model.loss(n_true_y, n_pred_y)
                grad = tape.gradient(n_loss_value, [
                    model.genre_converter_kernel,
                    model.genre_converter_bias,
                    model.creator_converter_kernel,
                    model.creator_converter_bias,
                    model.character_converter_kernel,
                    model.character_converter_bias,
                    model.item_embedding_converter,
                    model.predictor_layer1_kernel,
                    model.predictor_layer1_bias,
                    model.predictor_layer2_kernel,
                    model.predictor_layer2_bias,
                    model.predictor_layer3_kernel,
                    model.predictor_layer3_bias,
                    model.predictor_layer4_kernel
                ])
                user_preference_idx_grad = tape.gradient(n_loss_value,
                                                         model.hypernetwork.meta_user_embedding.meta_user_basis.trainable_weights)
                user_embed_grad = tape.gradient(n_loss_value,
                                                model.user_embedding.trainable_weights)
                item_embed_grad = tape.gradient(n_loss_value,
                                                model.item_embedding.trainable_weights)

            grad = [tf.clip_by_value(g, -1.0, 1.0) for g in grad]
            item_embed_grad = [tf.clip_by_value(g, -1.0, 1.0) for g in item_embed_grad]
            user_preference_idx_grad = [tf.clip_by_value(g, -1.0, 1.0) for g in user_preference_idx_grad]
            user_embed_grad = [tf.clip_by_value(g, -1.0, 1.0) for g in user_embed_grad]
            model.local_update(grad, learning_rate)
            optimizer.apply_gradients(zip(user_preference_idx_grad,
                                          model.hypernetwork.meta_user_embedding.meta_user_basis.trainable_weights))
            optimizer.apply_gradients(zip(user_embed_grad,
                                          model.user_embedding.trainable_weights))
            optimizer.apply_gradients(zip(item_embed_grad,
                                          model.item_embedding.trainable_weights))
            user_id = pos_sample[SUPPORT_SIZE:, 0]
            true_y = pos_sample[SUPPORT_SIZE:, 2].numpy()
            n_true_y = neg_sample[SUPPORT_SIZE:, 2]
            start_idx = 3
            n_user_id = neg_sample[SUPPORT_SIZE:, 0]
            genre_idx = pos_sample[SUPPORT_SIZE:, start_idx: start_idx + max_genre_size]
            n_genre_idx = neg_sample[SUPPORT_SIZE:, start_idx: start_idx + max_genre_size]
            start_idx = start_idx + max_genre_size
            character_idx = pos_sample[SUPPORT_SIZE:, start_idx: start_idx + max_character_size]
            n_character_idx = neg_sample[SUPPORT_SIZE:, start_idx: start_idx + max_character_size]
            start_idx = start_idx + max_character_size
            creator_idx = pos_sample[SUPPORT_SIZE:, start_idx: start_idx + max_creator_size]
            n_creator_idx = neg_sample[SUPPORT_SIZE:, start_idx: start_idx + max_creator_size]
            g_param_in_metanet = [
                model.hypernetwork.meta_genre_converter_kernel,
                model.hypernetwork.meta_character_converter_kernel,
                model.hypernetwork.meta_creator_converter_kernel,
                model.hypernetwork.meta_genre_converter_bias,
                model.hypernetwork.meta_character_converter_bias,
                model.hypernetwork.meta_creator_converter_bias,
                model.hypernetwork.meta_item_encoder_kernel,
                model.hypernetwork.meta_predictor_layer1_kernel,
                model.hypernetwork.meta_predictor_layer1_bias,
                model.hypernetwork.meta_predictor_layer2_kernel,
                model.hypernetwork.meta_predictor_layer2_bias,
                model.hypernetwork.meta_predictor_layer3_kernel,
                model.hypernetwork.meta_predictor_layer3_bias,
                model.hypernetwork.meta_predictor_layer4_kernel
            ]
            with tf.GradientTape(persistent=True) as tape:
                model.init_meta_embeddings(meta_user_id)
                pred_y, atten_s = model(user_id, genre_idx, creator_idx, character_idx)
                q_loss_value = model.loss(true_y, pred_y)
                n_pred_y, _ = model(n_user_id, n_genre_idx, n_creator_idx, n_character_idx)
                n_loss_value = model.loss(n_true_y, n_pred_y)
                grad = tape.gradient(q_loss_value,
                                     g_param_in_metanet)
                n_grad = tape.gradient(n_loss_value,
                                       g_param_in_metanet)
                user_pref_grad = tape.gradient(q_loss_value,
                                               model.hypernetwork.meta_user_embedding.meta_user_embeddings.trainable_weights)
                n_user_pref_grad = tape.gradient(n_loss_value,
                                                 model.hypernetwork.meta_user_embedding.meta_user_embeddings.trainable_weights)
            grad = [tf.clip_by_value(g, -1.0, 1.0) for g in grad]
            n_grad = [tf.clip_by_value(g, -1.0, 1.0) for g in n_grad]
            user_pref_grad = [tf.clip_by_value(g, -1.0, 1.0) for g in user_pref_grad]
            n_user_pref_grad = [tf.clip_by_value(g, -1.0, 1.0) for g in n_user_pref_grad]
            basis_optimizer.apply_gradients(zip(grad,
                                                g_param_in_metanet))
            basis_optimizer.apply_gradients(zip(n_grad,
                                                g_param_in_metanet))
            basis_optimizer.apply_gradients(zip(user_pref_grad,
                                                model.hypernetwork.meta_user_embedding.meta_user_embeddings.trainable_weights))
            basis_optimizer.apply_gradients(zip(n_user_pref_grad,
                                                model.hypernetwork.meta_user_embedding.meta_user_embeddings.trainable_weights))
            time_end = time.time()

            if ((idx > 0) and ((idx + train_user_cnt * epoch_num) % eval_batch) == 0):
                test_FedRL(model, optimizer, valid_dataset, valid_user_cnt,
                             SUPPORT_SIZE, QUERY_SIZE, "FedRL_Valid", tf_writer, epoch_num * train_user_cnt + idx)
                test_FedRL(model, optimizer, test_dataset, test_user_cnt,
                             SUPPORT_SIZE, QUERY_SIZE, "FedRL_Test", tf_writer, epoch_num * train_user_cnt + idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_config")
    args = parser.parse_args()
    model_cfg_filename = args.model_config
    with open("dataset_info.json", "r") as f_handle:
        dataset_cfg = json.load(f_handle)
    with open(model_cfg_filename, "r") as f_handle:
        model_cfg = json.load(f_handle)
    eval_batch = model_cfg['eval_batch']
    use_existing_items = model_cfg['use_existing_items']
    DATASET_IN_USE = dataset_cfg["DATASET"]
    CREATOR_NUM = dataset_cfg[DATASET_IN_USE]["CREATOR_NUM"]
    CHARACTER_NUM = dataset_cfg[DATASET_IN_USE]["CHARACTER_NUM"]
    GENRE_NUM = dataset_cfg[DATASET_IN_USE]["GENRE_NUM"]
    NEGATIVE_EXAMPLE_NUM = dataset_cfg[DATASET_IN_USE]["NEGATIVE_NUM"]
    SUPPORT_SIZE = dataset_cfg[DATASET_IN_USE]["SUPPORT_SIZE"]
    QUERY_SIZE = dataset_cfg[DATASET_IN_USE]["QUERY_SIZE"]
    if "max_epoch" in model_cfg:
        EPOCH_NUM = model_cfg["max_epoch"]
    else:
        EPOCH_NUM = model_cfg["max_steps"]
    USE_GPU = model_cfg["use_gpu"]
    os.environ["CUDA_VISIBLE_DEVICES"] = USE_GPU
    with open("enriched_ratings.pickle" + "." + DATASET_IN_USE, "rb") as f_handle:
        enriched_ratings = pickle.load(f_handle)
    user_cnt = 0
    valid_user = 0
    max_genre_size = 0
    max_creator_size = 0
    max_character_size = 0
    max_genre_id = 0
    max_creator_id = 0
    max_character_id = 0
    max_item_id = 0
    print('Prepare Dataset Statistics... Please Wait')
    for k, v in tqdm(enriched_ratings.items()):
        for vv in v:
            genre_size = len(vv[4])
            creator_size = len(vv[5])
            char_size = len(vv[6])
            item_id = int(vv[1])
            if item_id > max_item_id:
                max_item_id = item_id
            genre_int_id_list = map(lambda x: int(x), vv[4])
            creator_int_id_list = map(lambda x: int(x), vv[5])
            char_int_id_list = map(lambda x: int(x), vv[6])
            for _id in genre_int_id_list:
                if _id > max_genre_id:
                    if _id == 0:
                        print('crap')
                    max_genre_id = _id
            for _id in creator_int_id_list:
                if _id > max_creator_id:
                    max_creator_id = _id
            for _id in char_int_id_list:
                if _id > max_character_id:
                    max_character_id = _id
            if genre_size > max_genre_size:
                max_genre_size = genre_size
            if creator_size > max_creator_size:
                max_creator_size = creator_size
            if char_size > max_character_size:
                max_character_size = char_size
        if len(v) >= (SUPPORT_SIZE + QUERY_SIZE):
            valid_user += 1
        user_cnt += 1
    print('Valid User / Total User - {valid_user}/{user_cnt}'.format(
        valid_user=valid_user,
        user_cnt=user_cnt
    ))
    print('Data record statistics')
    print('Genre vector size, creator vector size, character vector size')
    print(max_genre_size, max_creator_size, max_character_size)
    print('Max item id, max genre id, max creator id, max character id (before filtering)')
    print(max_item_id, max_genre_id, max_creator_id, max_character_id)
    all_item_feature = defaultdict()
    with open('all_item_id.pickle.' + DATASET_IN_USE, "rb") as f_handle:
        all_item_id = pickle.load(f_handle)
    with open('all_item_mat.pickle.' + DATASET_IN_USE, "rb") as f_handle:
        all_item_mat = pickle.load(f_handle)
    with open('full_user_ids.pickle.' + DATASET_IN_USE, "rb") as f_handle:
        full_user_ids = pickle.load(f_handle)
    with open('all_item_feature.pickle.' + DATASET_IN_USE, "rb") as f_handle:
        all_item_feature = pickle.load(f_handle)
    full_user_ids = list(full_user_ids)
    if use_existing_items:
        train_user_cnt = int(len(full_user_ids) * 0.6)
        valid_user_cnt = int(len(full_user_ids) * 0.2)
        test_user_cnt = int(len(full_user_ids) * 0.2)
    else:
        if DATASET_IN_USE == 'ml-1m':
            train_user_cnt = int(len(full_user_ids) * 0.05)
            valid_user_cnt = int(len(full_user_ids) * 0.45)
            test_user_cnt = int(len(full_user_ids) * 0.45)
        if DATASET_IN_USE == 'Ciao':
            train_user_cnt = int(len(full_user_ids) * 0.05)
            valid_user_cnt = int(len(full_user_ids) * 0.45)
            test_user_cnt = int(len(full_user_ids) * 0.45)
        if DATASET_IN_USE == 'DM':
            train_user_cnt = int(len(full_user_ids) * 0.05)
            valid_user_cnt = int(len(full_user_ids) * 0.45)
            test_user_cnt = int(len(full_user_ids) * 0.45)
        else:
            train_user_cnt = int(len(full_user_ids) * 0.05)
            valid_user_cnt = int(len(full_user_ids) * 0.45)
            test_user_cnt = int(len(full_user_ids) * 0.45)
    train_user_ids = full_user_ids[:train_user_cnt]
    valid_user_ids = full_user_ids[train_user_cnt: train_user_cnt + valid_user_cnt]
    test_user_ids = full_user_ids[train_user_cnt + valid_user_cnt: train_user_cnt + valid_user_cnt + test_user_cnt]
    train_movie_id = set()
    print("Preparing Items in Training Set...")
    for u_ids in tqdm(train_user_ids):
        u_rec = enriched_ratings[str(u_ids)]
        for rec in u_rec[: SUPPORT_SIZE + QUERY_SIZE]:
            movie_id = int(rec[1])
            train_movie_id.add(movie_id)
    print("Preparing Items in Validation Set...")
    valid_user_ids_ = list()
    for u_ids in tqdm(valid_user_ids):
        u_rec = enriched_ratings[str(u_ids)]
        valid_rec_cnt = 0
        for rec in u_rec:
            if DATASET_IN_USE != 'ml-1m':
                movie_id = int(rec[1])
                if use_existing_items:
                    if movie_id in train_movie_id:
                        valid_rec_cnt += 1
                else:
                    if movie_id not in train_movie_id:
                        valid_rec_cnt += 1
            else:
                valid_rec_cnt += 1
        if valid_rec_cnt >= (SUPPORT_SIZE + QUERY_SIZE):
            valid_user_ids_.append(u_ids)
    print("Preparing Items in Test Set...")
    test_user_ids_ = list()
    for u_ids in tqdm(test_user_ids):
        u_rec = enriched_ratings[str(u_ids)]
        valid_rec_cnt = 0
        for rec in u_rec:
            movie_id = int(rec[1])
            if use_existing_items:
                if movie_id in train_movie_id:
                    valid_rec_cnt += 1
            else:
                if movie_id not in train_movie_id:
                    valid_rec_cnt += 1
        if valid_rec_cnt >= (SUPPORT_SIZE + QUERY_SIZE):
            test_user_ids_.append(u_ids)
    valid_user_ids = valid_user_ids_
    test_user_ids = test_user_ids_
    valid_user_cnt = len(valid_user_ids)
    test_user_cnt = len(test_user_ids)
    if use_existing_items:
        print("Training on existing item -> existing item")
    print(f"Training User #: {train_user_cnt}."
          f"Valid User #: {valid_user_cnt}."
          f"Test User #: {test_user_cnt}."
          )
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    train_dataset = tf.data.Dataset.from_generator(
        generator=lambda: get_dataset(train_user_ids, enriched_ratings, all_item_id,
                                      all_item_feature, max_genre_size, max_creator_size, max_character_size,
                                      train_movie_id, NEGATIVE_EXAMPLE_NUM, SUPPORT_SIZE, QUERY_SIZE),
        output_types=(tf.float32, tf.float32)
    )
    train_dataset = train_dataset.prefetch(10)
    valid_dataset = tf.data.Dataset.from_generator(
        generator=lambda: get_dataset(valid_user_ids, enriched_ratings, all_item_id,
                                      all_item_feature, max_genre_size, max_creator_size, max_character_size,
                                      train_movie_id, NEGATIVE_EXAMPLE_NUM, SUPPORT_SIZE, QUERY_SIZE),
        output_types=(tf.float32, tf.float32)
    )
    valid_dataset = valid_dataset.prefetch(10)
    test_dataset = tf.data.Dataset.from_generator(
        generator=lambda: get_dataset(test_user_ids, enriched_ratings, all_item_id,
                                      all_item_feature, max_genre_size, max_creator_size, max_character_size,
                                      train_movie_id, NEGATIVE_EXAMPLE_NUM, SUPPORT_SIZE, QUERY_SIZE),
        output_types=(tf.float32, tf.float32)
    )
    test_dataset = test_dataset.prefetch(10)
    model_name = model_cfg["model_name"]
    writer_name = DATASET_IN_USE + "/" + model_name + "_" + str(time.time())
    tf_writer = tf.summary.create_file_writer("./log/" + writer_name)
    with open(writer_name + "_config.log", "w") as conf_writer:
        json.dump(model_cfg, conf_writer)
    if model_name == "FedRL":
        model = FedRLRecommender(dataset_cfg[DATASET_IN_USE], model_cfg)
        learning_rate = model_cfg["alpha"]
        meta_learning_rate = model_cfg["beta"]
        basis_learning_rate = model_cfg["gamma"]
        basis_optimizer = tf.keras.optimizers.Adam(learning_rate=basis_learning_rate)
        optimizer = tf.keras.optimizers.Adam(learning_rate=meta_learning_rate)
        train_FedRL(model, optimizer, DATASET_IN_USE, basis_optimizer, train_dataset, valid_dataset, test_dataset,
                      learning_rate,
                      train_user_cnt, valid_user_cnt, test_user_cnt,
                      SUPPORT_SIZE, QUERY_SIZE, EPOCH_NUM, tf_writer)
