import tensorflow as tf
from tensorflow import keras
import numpy as np
from copy import deepcopy
#client
class UserMetaEmbeddings(keras.layers.Layer):
    def __init__(self, dataset_cfg, model_cfg):
        super(UserMetaEmbeddings, self).__init__()
        user_num = dataset_cfg["USER_NUM"]
        self.meta_user_basis = keras.layers.Embedding(
            input_dim = user_num,
            output_dim = model_cfg["interest_basis_num"]
        )
        self.meta_user_embeddings = keras.models.Sequential(
            [
                tf.keras.Input(shape=(model_cfg["interest_basis_num"])),
                keras.layers.Dense(units=model_cfg["interest_basis_dim"], use_bias=False)
            ]
        )
    def call(self, user_idx):
        user_meta_embedding_idx = self.meta_user_basis(user_idx)
        user_meta_embedding_idx = tf.reshape(user_meta_embedding_idx, [1, -1])
        user_meta_embedding = self.meta_user_embeddings(user_meta_embedding_idx)
        return user_meta_embedding
class MetaNet(keras.layers.Layer):
    def __init__(self, dataset_cfg, model_cfg):
        super(MetaNet, self).__init__()
        self.meta_user_embedding = UserMetaEmbeddings(dataset_cfg, model_cfg)
        meta_user_embedding_size = model_cfg["interest_basis_dim"]
        self.embedding_size = model_cfg["embedding_size"]
        embedding_size = self.embedding_size
        if dataset_cfg['USER_NUM'] == :
            self.relu_func = tf.keras.layers.LeakyReLU(name="FedRL_MetaNet_LeakyReLU")
        else:
            self.relu_func = tf.keras.layers.ReLU(name='FedRL_MetaNet_ReLU')
        self.meta_genre_converter_kernel = self.add_weight(
            name = "meta_genre_converter_kernel",
            shape=[meta_user_embedding_size, embedding_size ** 2]
        )
        self.meta_character_converter_kernel = self.add_weight(
            name = "meta_character_converter_kernel",
            shape=[meta_user_embedding_size, embedding_size ** 2]
        )
        self.meta_creator_converter_kernel = self.add_weight(
            name = "meta_creator_converter_kernel",
            shape=[meta_user_embedding_size, embedding_size ** 2]
        )
        self.meta_genre_converter_bias = self.add_weight(
            name = "meta_genre_converter_bias",
            shape=[meta_user_embedding_size, embedding_size]
        )
        self.meta_character_converter_bias = self.add_weight(
            name = "meta_character_converter_bias",
            shape=[meta_user_embedding_size, embedding_size]
        )
        self.meta_creator_converter_bias = self.add_weight(
            name = "meta_creator_converter_bias",
            shape=[meta_user_embedding_size, embedding_size]
        )
        self.meta_item_encoder_kernel = self.add_weight(
            name = "meta_item_encoder_kernel",
            shape=[meta_user_embedding_size, embedding_size * 3 * embedding_size]
        )
        self.meta_predictor_layer1_kernel = self.add_weight(
            name = "meta_predictor_layer1_kernel",
            shape=[meta_user_embedding_size, embedding_size * 2 * embedding_size]
        )
        self.meta_predictor_layer1_bias = self.add_weight(
            name = "meta_predictor_layer1_bias",
            shape=[meta_user_embedding_size, embedding_size]
        )
        self.meta_predictor_layer2_kernel = self.add_weight(
            name = "meta_predictor_layer2_kernel",
            shape=[meta_user_embedding_size, embedding_size * embedding_size]
        )
        self.meta_predictor_layer2_bias = self.add_weight(
            name = "meta_predictor_layer2_bias",
            shape=[meta_user_embedding_size, embedding_size]
        )
        self.meta_predictor_layer3_kernel = self.add_weight(
            name = "meta_predictor_layer3_kernel",
            shape=[meta_user_embedding_size, embedding_size * embedding_size]
        )
        self.meta_predictor_layer3_bias = self.add_weight(
            name = "meta_predictor_layer3_bias",
            shape=[meta_user_embedding_size, embedding_size]
        )
        self.meta_predictor_layer4_kernel = self.add_weight(
            name = "meta_predictor_layer4_kernel",
            shape=[meta_user_embedding_size, embedding_size]
        )
    def call(self, user_idx):
        user_interest_embedding = self.meta_user_embedding(user_idx)
        embedding_size = self.embedding_size
        genre_converter_kernel = tf.linalg.matmul(user_interest_embedding, 
                                            self.meta_genre_converter_kernel)
        genre_converter_kernel = tf.reshape(genre_converter_kernel,
                                            [embedding_size, embedding_size])
        character_converter_kernel = tf.linalg.matmul(user_interest_embedding,
                                           self.meta_character_converter_kernel)
        character_converter_kernel = self.relu_func(character_converter_kernel)

        character_converter_kernel = tf.reshape(character_converter_kernel,
                                           [embedding_size, embedding_size])
        
        creator_converter_kernel = tf.linalg.matmul(user_interest_embedding,
                                            self.meta_creator_converter_kernel)
        creator_converter_kernel = self.relu_func(creator_converter_kernel)
        creator_converter_kernel = tf.reshape(creator_converter_kernel,
                                            [embedding_size, embedding_size])
        genre_converter_bias = tf.linalg.matmul(user_interest_embedding,
                                            self.meta_genre_converter_bias)
        genre_converter_bias = self.relu_func(genre_converter_bias)
        genre_converter_bias = tf.reshape(genre_converter_bias, [1, embedding_size])
        character_converter_bias = tf.linalg.matmul(user_interest_embedding,
                                            self.meta_character_converter_bias)
        character_converter_bias = self.relu_func(character_converter_bias)
        character_converter_bias = tf.reshape(character_converter_bias, [1, embedding_size])
        creator_converter_bias = tf.linalg.matmul(user_interest_embedding,
                                            self.meta_creator_converter_bias)
        creator_converter_bias = self.relu_func(creator_converter_bias)
        creator_converter_bias = tf.reshape(creator_converter_bias, [1, embedding_size])
        meta_item_encoder_kernel = tf.linalg.matmul(user_interest_embedding,
                                                self.meta_item_encoder_kernel)
        meta_item_encoder_kernel = self.relu_func(meta_item_encoder_kernel)
        meta_item_encoder_kernel = tf.reshape(meta_item_encoder_kernel,
                                             [embedding_size * 3, embedding_size])
        meta_predictor_layer1_kernel = tf.linalg.matmul(user_interest_embedding,
                                                self.meta_predictor_layer1_kernel)
        meta_predictor_layer1_kernel = self.relu_func(meta_predictor_layer1_kernel)
        meta_predictor_layer1_kernel = tf.reshape(meta_predictor_layer1_kernel,
                                            [embedding_size * 2, embedding_size])
        meta_predictor_layer1_bias = tf.linalg.matmul(user_interest_embedding,
                                                self.meta_predictor_layer1_bias)
        meta_predictor_layer1_bias = self.relu_func(meta_predictor_layer1_bias)
        meta_predictor_layer1_bias = tf.reshape(meta_predictor_layer1_bias,
                                            [1, embedding_size])
        meta_predictor_layer2_kernel = tf.linalg.matmul(user_interest_embedding,
                                                self.meta_predictor_layer2_kernel)
        meta_predictor_layer2_kernel = self.relu_func(meta_predictor_layer2_kernel)
        meta_predictor_layer2_kernel = tf.reshape(meta_predictor_layer2_kernel,
                                            [embedding_size, embedding_size])
        meta_predictor_layer2_bias = tf.linalg.matmul(user_interest_embedding,
                                                self.meta_predictor_layer2_bias)
        meta_predictor_layer2_bias = self.relu_func(meta_predictor_layer2_bias)
        meta_predictor_layer2_bias = tf.reshape(meta_predictor_layer2_bias,
                                            [1, embedding_size])
        meta_predictor_layer3_kernel = tf.linalg.matmul(user_interest_embedding,
                                                self.meta_predictor_layer3_kernel)
        meta_predictor_layer3_kernel = self.relu_func(meta_predictor_layer3_kernel)
        meta_predictor_layer3_kernel = tf.reshape(meta_predictor_layer3_kernel,
                                            [embedding_size, embedding_size])
        meta_predictor_layer3_bias = tf.linalg.matmul(user_interest_embedding,
                                                self.meta_predictor_layer3_bias)
        meta_predictor_layer3_bias = self.relu_func(meta_predictor_layer3_bias)
        meta_predictor_layer3_bias = tf.reshape(meta_predictor_layer3_bias,
                                            [1, embedding_size])
        meta_predictor_layer4_kernel = tf.linalg.matmul(user_interest_embedding,
                                                self.meta_predictor_layer4_kernel)
        meta_predictor_layer4_kernel = self.relu_func(meta_predictor_layer4_kernel)
        meta_predictor_layer4_kernel = tf.reshape(meta_predictor_layer4_kernel,
                                            [embedding_size, 1])
        return (genre_converter_kernel, genre_converter_bias,
                character_converter_kernel, character_converter_bias,
                creator_converter_kernel, creator_converter_bias,
                meta_item_encoder_kernel,
                meta_predictor_layer1_kernel, meta_predictor_layer1_bias,
                meta_predictor_layer2_kernel, meta_predictor_layer2_bias,
                meta_predictor_layer3_kernel, meta_predictor_layer3_bias,
                meta_predictor_layer4_kernel)
class UserEmbeddings(keras.layers.Layer):
    def __init__(self, dataset_cfg, model_cfg):
        super(UserEmbeddings, self).__init__()
        self.user_embeddings = keras.layers.Embedding(
            input_dim = dataset_cfg["USER_NUM"],
            output_dim = model_cfg["embedding_size"],
            name = "user_embed_layer"
        )
    def call(self, user_embed_idx):
        return self.user_embeddings(user_embed_idx)
class ItemEmbeddings(keras.layers.Layer):
    def __init__(self, dataset_cfg, model_cfg):
        super(ItemEmbeddings, self).__init__()
        genre_num = dataset_cfg["GENRE_NUM"]
        creator_num = dataset_cfg["CREATOR_NUM"]
        character_num = dataset_cfg["CHARACTER_NUM"]
        embedding_size = model_cfg["embedding_size"]
        self.genre_embeddings = keras.layers.Embedding(name="item_genre_embedding",
                                                       embeddings_initializer='glorot_uniform',
                                                       input_dim=genre_num,
                                                       output_dim=embedding_size)
        self.creator_embeddings = keras.layers.Embedding(name="item_creator_embedding",
                                                         embeddings_initializer='glorot_uniform',
                                                         input_dim=creator_num,
                                                         output_dim=embedding_size)
        self.character_embeddings = keras.layers.Embedding(name="item_character_embedding",
                                                           embeddings_initializer='glorot_uniform',
                                                           input_dim=character_num,
                                                           output_dim=embedding_size)
    def call(self, genre_vec, creator_vec, character_vec):
        genre_embedding = self.genre_embeddings(genre_vec)
        genre_embedding = tf.math.reduce_sum(genre_embedding, axis=-2)
        creator_embedding = tf.math.reduce_sum(self.creator_embeddings(creator_vec), axis=-2)
        character_embedding = tf.math.reduce_sum(self.character_embeddings(character_vec), axis=-2)
        return genre_embedding, creator_embedding, character_embedding
class FedRLRecommender(keras.models.Model):
    def __init__(self, dataset_cfg, model_cfg):
        super(FedRLRecommender, self).__init__()
        self.embedding_size = model_cfg["embedding_size"]
        self.item_embedding = ItemEmbeddings(dataset_cfg, model_cfg)
        self.user_embedding = UserEmbeddings(dataset_cfg, model_cfg)
        self.hypernetwork = MetaNet(dataset_cfg, model_cfg)
        self.interest_attention = tf.keras.layers.Attention(
            use_scale = False
        )
        if dataset_cfg['USER_NUM'] == :
            self.relu_func = tf.keras.layers.LeakyReLU(name="FedRL_MetaNet_LeakyReLU")
            tf.print('---------------------Detecting MovieLens-1M Dataset. Use LeakyReLU-------------------------')
        else:
            self.relu_func = tf.keras.layers.ReLU(name='FedRL_MetaNet_ReLU')
        self.genre_converter_kernel = None
        self.genre_converter_bias = None
        self.creator_converter_kernel = None
        self.creator_converter_bias = None
        self.character_converter_kernel = None
        self.character_converter_bias = None
        self.item_embedding_converter = None
        self.predictor_layer1_kernel = None
        self.predictor_layer1_bias = None
        self.predictor_layer2_kernel = None
        self.predictor_layer2_bias = None
        self.predictor_layer3_kernel = None
        self.predictor_layer3_bias = None
        self.predictor_layer4_kernel = None

    def init_meta_embeddings(self, user_idx):
        (genre_converter_kernel, genre_converter_bias,
        character_converter_kernel, character_converter_bias,
        creator_converter_kernel, creator_converter_bias,
        item_encoder_kernel,
        predictor_layer1_kernel,
        predictor_layer1_bias,
        predictor_layer2_kernel,
        predictor_layer2_bias,
        predictor_layer3_kernel,
        predictor_layer3_bias,
        predictor_layer4_kernel) = self.hypernetwork(user_idx)
        self.genre_converter_kernel = genre_converter_kernel
        self.genre_converter_bias = genre_converter_bias
        self.creator_converter_kernel = creator_converter_kernel
        self.creator_converter_bias = creator_converter_bias
        self.character_converter_kernel = character_converter_kernel
        self.character_converter_bias = character_converter_bias
        self.item_embedding_converter = item_encoder_kernel
        self.predictor_layer1_kernel = predictor_layer1_kernel
        self.predictor_layer1_bias = predictor_layer1_bias
        self.predictor_layer2_kernel = predictor_layer2_kernel
        self.predictor_layer2_bias = predictor_layer2_bias
        self.predictor_layer3_kernel = predictor_layer3_kernel
        self.predictor_layer3_bias = predictor_layer3_bias
        self.predictor_layer4_kernel = predictor_layer4_kernel
    def local_update(self, grad, lr):
        self.genre_converter_kernel = self.genre_converter_kernel - lr * grad[0]
        self.genre_converter_bias = self.genre_converter_bias - lr * grad[1]
        self.creator_converter_kernel = self.creator_converter_kernel - lr * grad[2]
        self.creator_converter_bias = self.creator_converter_bias - lr * grad[3]
        self.character_converter_kernel = self.character_converter_kernel - lr * grad[4]
        self.character_converter_bias = self.character_converter_bias - lr * grad[5]
        self.item_embedding_converter = self.item_embedding_converter - lr * grad[6]
        self.predictor_layer1_kernel = self.predictor_layer1_kernel - lr * grad[7]
        self.predictor_layer1_bias = self.predictor_layer1_bias - lr * grad[8]
        self.predictor_layer2_kernel = self.predictor_layer2_kernel - lr * grad[9]
        self.predictor_layer2_bias = self.predictor_layer2_bias - lr * grad[10]
        self.predictor_layer3_kernel = self.predictor_layer3_kernel - lr * grad[11]
        self.predictor_layer3_bias = self.predictor_layer3_bias - lr * grad[12]
        self.predictor_layer4_kernel = self.predictor_layer4_kernel - lr * grad[13]
    def call(self, user_id, genre_idx, creator_idx, character_idx, dbg=False):
        user_idx = tf.cast(user_id, dtype=tf.int32)
        user_embed = self.user_embedding(user_idx)
        genre_embed, creator_embed, character_embed = self.item_embedding(genre_idx, creator_idx,
                                                            character_idx)
        genre_embed = tf.matmul(genre_embed, self.genre_converter_kernel)
        genre_embed = genre_embed + self.genre_converter_bias
        genre_embed = self.relu_func(genre_embed)
        creator_embed = tf.matmul(creator_embed, self.creator_converter_kernel)
        creator_embed = creator_embed + self.creator_converter_bias
        creator_embed = self.relu_func(creator_embed)
        character_embed = tf.matmul(character_embed, self.character_converter_kernel)
        character_embed = character_embed + self.character_converter_bias
        character_embed = self.relu_func(character_embed)
        q = tf.expand_dims(user_embed, axis=1)
        v = tf.stack([genre_embed, creator_embed, character_embed], axis=1)
        att_item_embedding, att_score = self.interest_attention([q, v, v], 
                                            return_attention_scores = True)
        v = tf.squeeze(v)
        naive_item_embedding = tf.concat([genre_embed, creator_embed, character_embed], axis=1)
        item_embedding = tf.matmul(naive_item_embedding, self.item_embedding_converter)
        att_item_embedding = tf.squeeze(att_item_embedding)
        att_score = tf.squeeze(att_score)
        mix_item_embedding = item_embedding + att_item_embedding
        pred_embedding = tf.concat([user_embed, mix_item_embedding], axis=1)
        pred_embedding = tf.matmul(pred_embedding, self.predictor_layer1_kernel)
        pred_embedding = pred_embedding + self.predictor_layer1_bias
        pred_embedding = self.relu_func(pred_embedding)
        pred_embedding = tf.matmul(pred_embedding, self.predictor_layer2_kernel)
        pred_embedding = pred_embedding + self.predictor_layer2_bias
        pred_embedding = self.relu_func(pred_embedding)
        pred_embedding = tf.matmul(pred_embedding, self.predictor_layer3_kernel)
        pred_embedding = pred_embedding + self.predictor_layer3_bias
        pred_embedding = self.relu_func(pred_embedding)
        pred_y = tf.matmul(pred_embedding, self.predictor_layer4_kernel)
        return pred_y, att_score
    def loss(self, true_y, pred_y):
        loss_value = (true_y - pred_y) ** 2
        return tf.math.reduce_mean(loss_value)
