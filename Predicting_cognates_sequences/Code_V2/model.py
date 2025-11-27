import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, BatchNormalization, Activation,
    Add, Concatenate, Dense, Dropout, Layer
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
import keras_tuner as kt


# -------------------------
# FOCAL LOSS (opcional)
# -------------------------
def focal_loss(gamma=2., alpha=.25):
    """
    Implementa a Focal Loss para lidar com desequilíbrio de classes.
    """
    def loss(y_true, y_pred):
        # Evita log(0)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return alpha * tf.pow((1 - p_t), gamma) * bce
    return loss


# -------------------------
# ATTENTION POOLING (CORRIGIDO: Agora como uma classe Layer)
# -------------------------
class AttentionPoolingLayer(Layer):
    """
    Camada customizada para realizar Attention Pooling de sequências.
    Isto resolve o problema de KerasTensor com tf.reduce_sum e a criação
    múltipla de variáveis no Keras Tuner.
    """
    def __init__(self, **kwargs):
        super(AttentionPoolingLayer, self).__init__(**kwargs)
        # As camadas com variáveis (Dense) são criadas uma única vez no __init__
        self.score_dense = Dense(1, activation="tanh", name="att_score_dense")
        self.score_softmax = Activation("softmax", name="att_softmax")

    def call(self, x):
        # 1. Calcula os scores de atenção
        score = self.score_dense(x) 
        score = self.score_softmax(score)
        
        # 2. Aplica o peso (KerasTensor * KerasTensor é seguro)
        weighted_x = score * x
        
        # 3. Redução (tf.reduce_sum é seguro dentro do método call de uma Layer)
        return tf.reduce_sum(weighted_x, axis=1)


# -------------------------
# MULTI-KERNEL BLOCK (multi-scale CNN)
# -------------------------
def multi_kernel_block(x, hp_filters):
    """Convoluções em paralelo com diferentes tamanhos de kernel para multi-escala."""
    b1 = Conv1D(hp_filters, 5, padding="same", activation="relu")(x)
    b2 = Conv1D(hp_filters, 11, padding="same", activation="relu")(x)
    b3 = Conv1D(hp_filters, 21, padding="same", activation="relu")(x)
    return Concatenate()([b1, b2, b3])


# -------------------------
# RESIDUAL DILATED BLOCK
# -------------------------
def residual_dilated_block(x, hp_filters, dilation_rate):
    """Bloco Residual com Convoluções Dilatadas."""
    input_tensor = x

    out_filters = x.shape[-1] 

    y = Conv1D(out_filters, 11, padding="same",
               dilation_rate=dilation_rate,
               kernel_initializer='he_normal')(x)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    y = Conv1D(out_filters, 11, padding="same",
               dilation_rate=dilation_rate,
               kernel_initializer='he_normal')(y)

    y = Add()([y, input_tensor])
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y


# -------------------------
# MAIN MODEL WITH HP
# -------------------------
def build_model(hp):

    # Hyperparams a serem otimizados pelo Keras Tuner
    hp_filters = hp.Choice("filters", values=[32, 64, 128])
    hp_dense = hp.Choice("dense_units", values=[64, 128, 256, 512])
    hp_dropout = hp.Float("dropout_rate", 0.1, 0.6, step=0.1)
    hp_l2 = hp.Choice("l2_reg", values=[1e-4, 1e-3, 1e-2])
    hp_lr = hp.Choice("learning_rate", values=[1e-4, 5e-4, 1e-3, 5e-3]) 
    hp_embed = hp.Choice("embed_dim", values=[4, 8, 16, 32])
    hp_batch_size = hp.Choice("batch_size", values=[16, 32, 64, 128])

    SEQ_LEN = 201
    VOCAB = 5  # 4 bases + 1 para 'N'

    # -------------------------
    # INPUTS & EMBEDDING
    # -------------------------
    donor_input = Input(shape=(SEQ_LEN,), name="donor_input")
    acceptor_input = Input(shape=(SEQ_LEN,), name="acceptor_input")
    
    donor = Embedding(VOCAB, hp_embed, name="donor_embed")(donor_input)
    acceptor = Embedding(VOCAB, hp_embed, name="acceptor_embed")(acceptor_input)
    
    # Instancia a camada de Attention Pooling
    attention_layer = AttentionPoolingLayer()

    # -------------------------
    # DONOR BRANCH
    # -------------------------
    xD = multi_kernel_block(donor, hp_filters)
    for rate in [1, 2, 4, 8]:
        xD = residual_dilated_block(xD, hp_filters, rate)
        
    # USO DA CLASSE Layer: Chamada como uma camada Keras
    donor_feat = attention_layer(xD)

    # -------------------------
    # ACCEPTOR BRANCH
    # -------------------------
    xA = multi_kernel_block(acceptor, hp_filters)

    for rate in [1, 2, 4, 8]:
        xA = residual_dilated_block(xA, hp_filters, dilation_rate=rate)
        
    # USO DA CLASSE Layer: Chamada como uma camada Keras
    acceptor_feat = attention_layer(xA)

    # -------------------------
    # FEATURE FUSION + GATING
    # -------------------------
    fused = Concatenate()([donor_feat, acceptor_feat])

    # Gating
    gate = Dense(fused.shape[-1], activation="sigmoid", name="fusion_gate")(fused)
    fused = fused * gate 

    # -------------------------
    # CLASSIFICATION HEAD
    # -------------------------
    y = Dense(hp_dense, activation="relu",
              kernel_regularizer=l2(hp_l2), name="dense_1")(fused)
    y = Dropout(hp_dropout)(y)

    y = Dense(hp_dense // 2, activation="relu",
              kernel_regularizer=l2(hp_l2), name="dense_2")(y)
    y = Dropout(hp_dropout)(y)

    output = Dense(1, activation="sigmoid", name="output")(y)

    model = Model(inputs=[donor_input, acceptor_input], outputs=output)

    # Compile
    model.compile(
        optimizer=AdamW(learning_rate=hp_lr, weight_decay=hp_l2),
        loss=focal_loss(gamma=2., alpha=0.25), 
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model