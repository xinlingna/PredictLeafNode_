from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate, Activation
from keras.optimizers import Adam
from keras.optimizers import AdamW
from keras.losses import KLDivergence
from keras.metrics import top_k_categorical_accuracy
import keras.backend as K
import tensorflow as tf

import tensorflow as tf
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers.schedules import ExponentialDecay

def custom_top_k_categorical_accuracy(y_true, y_pred, k=10):

    # 获取预测概率的 top-k 索引
    top_k_pred = tf.math.top_k(y_pred, k=k).indices
    
    # 获取真实标签的类别索引
    top_k_true = tf.math.top_k(y_true, k=k).indices
    
    # 检查真实标签是否在预测的 top-k 类别中
    correct = tf.reduce_any(
        tf.equal(tf.expand_dims(top_k_pred, -1), tf.expand_dims(top_k_true, 1)), 
        axis=-1
    )
    
    # 计算准确率：正确的数量除以总样本数
    return tf.reduce_mean(tf.cast(correct, tf.float32))


def build_model(input_query_dim=128, input_sim_dim=300, hidden_units=1024, output_dim=300,top_k=10):
    # 输入：查询向量
    input_query = Input(shape=(input_query_dim,), name="query_input")
    x_q = Dense(input_query_dim, kernel_initializer='he_uniform')(input_query)
    x_q = BatchNormalization()(x_q)
    x_q = Activation('relu')(x_q)
    x_q = Dropout(0.3)(x_q)

    # 输入：相似度向量
    input_sim = Input(shape=(input_sim_dim,), name="sim_input")
    x_s = Dense(input_sim_dim, kernel_initializer='he_uniform')(input_sim)
    x_s = BatchNormalization()(x_s)
    x_s = Activation('relu')(x_s)
    x_s = Dropout(0.3)(x_s)

    # 融合
    x = Concatenate()([x_q, x_s])
    
    # 融合后多层MLP，渐进式
    x = Dense(hidden_units, kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    x = Dense(hidden_units, kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    x = Dense(hidden_units, kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    x = Dense(hidden_units // 2, kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)


    # 输出层
    output = Dense(output_dim, kernel_initializer='he_uniform')(x)
    output = Activation('softmax')(output)  # 输出为概率分布

    model = Model(inputs=[input_query, input_sim], outputs=output)
    
    # 动态学习率设置（初始值1e-4，每10个epoch衰减10%）
    """ lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=100,
        decay_rate=0.95) """
    lr_schedule=3e-4
    model.compile(
        # loss=KLDivergence(), 
        loss='categorical_crossentropy',
        
        # optimizer=Adam(learning_rate=lr_schedule), 
        optimizer=AdamW(learning_rate=lr_schedule,weight_decay=1e-5), 
        metrics=['accuracy', 'KLDivergence',lambda y_true, y_pred: custom_top_k_categorical_accuracy(y_true, y_pred, top_k)]
    )
    return model
