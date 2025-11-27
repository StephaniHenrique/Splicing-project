import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, GlobalMaxPooling1D, Concatenate, Dense, Activation,
    BatchNormalization, Add, Dropout
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
#Keras tuner to random search
import keras_tuner as kt 

N_BASES = 4 
SEQUENCE_LENGTH = 201 


def build_model(hp):
    
    #Hyperparameters to convolutional layers 
    hp_filters = hp.Int('filters', min_value=16, max_value=64, step=16)
    hp_kernel_size = hp.Int('kernel_size', min_value=5, max_value=15, step=5)
    hp_l2_reg = hp.Choice('l2_reg', values=[1e-4, 1e-3, 1e-2])
    hp_conv_dropout_rate = hp.Float('conv_dropout_rate', min_value=0.0, max_value=0.3, step=0.1)
    
    #Hyperparameters to dense layer
    hp_dense_units_1 = hp.Int('dense_units_1', min_value=128, max_value=512, step=128)
    hp_dense_units_2 = hp.Int('dense_units_2', min_value=32, max_value=128, step=32)
    hp_dense_dropout_rate = hp.Float('dense_dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    
    #Hyperparameters to optimize 
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 5e-3])

    #--- AUX FUNCTIONS ---

    def create_res_block_hp(x, rate):
        """Residual Block with Dilation and Dynamic Hyperparameters."""
        input_tensor = x
        
        #First Conv1D
        y = Conv1D(
            filters=hp_filters, 
            kernel_size=hp_kernel_size, 
            dilation_rate=rate, 
            padding='same', 
            kernel_initializer='he_normal',
        )(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        # Second Conv1D
        y = Conv1D(
            filters=hp_filters, 
            kernel_size=hp_kernel_size, 
            dilation_rate=rate, 
            padding='same', 
            kernel_initializer='he_normal',
        )(y)
        
        #Residual conection and normalization 
        y = Add()([y, input_tensor])
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        
        #Dropout
        if hp_conv_dropout_rate > 0:
            y = Dropout(hp_conv_dropout_rate)(y)
        
        return y

    def initial_conv_block_hp(x):
        
        x = Conv1D(
            filters=hp_filters, 
            kernel_size=hp_kernel_size, 
            padding='same', 
            kernel_initializer='he_normal', 
            kernel_regularizer=l2(hp_l2_reg) #Using L2
        )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    #--- MAIN ARCHITECTURE ---

    input_donor = Input(shape=(SEQUENCE_LENGTH, N_BASES), name='donor_input')
    input_acceptor = Input(shape=(SEQUENCE_LENGTH, N_BASES), name='acceptor_input')

    #DONOR BRANCH
    x_donor = initial_conv_block_hp(input_donor)
    dilation_rates = [1, 2, 4, 8] 
    for rate in dilation_rates:
        x_donor = create_res_block_hp(x_donor, rate=rate)
    features_donor = GlobalMaxPooling1D(name='donor_features')(x_donor) 
    
    #ACCEPTOR BRANCH
    x_acceptor = initial_conv_block_hp(input_acceptor)
    for rate in dilation_rates:
        x_acceptor = create_res_block_hp(x_acceptor, rate=rate)
    features_acceptor = GlobalMaxPooling1D(name='acceptor_features')(x_acceptor) 

    #FUSION
    fused_features = Concatenate(name='fused_features')([features_donor, features_acceptor])

    #CLASSIFICATION HEAD
    y = Dense(hp_dense_units_1, activation='relu')(fused_features)
    y = Dropout(hp_dense_dropout_rate)(y) 
    y = Dense(hp_dense_units_2, activation='relu')(y)
    y = Dropout(hp_dense_dropout_rate)(y) 
    
    output = Dense(1, activation='sigmoid', name='compatibility_output')(y)

    model = Model(inputs=[input_donor, input_acceptor], outputs=output)
    
    #COMPILATION
    model.compile(
        optimizer=AdamW(learning_rate=hp_learning_rate, weight_decay=hp_l2_reg), 
        loss='binary_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model


#Tuner config (Random Search)
tuner = kt.RandomSearch(
    build_model,
    objective='val_auc', 
    max_trials=20, 
    executions_per_trial=1, 
    directory='hpo_results', 
    project_name='spliceai_fidelity_search'
)

#Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=20, 
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=0
)

#We can update here to test different strategies (ex: without early stopping)
callbacks_list = [early_stopping, reduce_lr]

#--- Execution ---

try:
    #Loading data
    X_donor = np.load('X_donor_encoded.npy')
    X_acceptor = np.load('X_acceptor_encoded.npy')
    y = np.load('y_labels.npy')

    print(f"Shape dos Dados do Donor: {X_donor.shape}")
    print(f"Shape dos Dados do Acceptor: {X_acceptor.shape}")

 
    #Split data
    X_D_train, X_D_test, X_A_train, X_A_test, y_train, y_test = train_test_split(
        X_donor, X_acceptor, y, test_size=0.1, random_state=42
    )

    #Class Weight 
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw = dict(enumerate(cw))

    
    #RANDOM SEARCH
    tuner.search(
        [X_D_train, X_A_train], 
        y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.1,
        class_weight=cw,
        callbacks=callbacks_list,
        verbose=1
    )

    #Evaluating best value 
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)

    print("\n--- Hyperparameters ---")
    print(best_hps.values)
    
    
    print("\n--- Retraining best value ---")
   
    best_model.fit(
        [X_D_train, X_A_train], 
        y_train,
        epochs=150, 
        batch_size=64, 
        validation_split=0.1, 
        shuffle=True, 
        class_weight=cw,
        callbacks=callbacks_list #(EarlyStopping, ReduceLROnPlateau)
    )

    #Final evaluation 
    loss, accuracy, auc = best_model.evaluate([X_D_test, X_A_test], y_test, verbose=0)
    print(f"\nResultado do Melhor Modelo no Teste (Sequence Length {SEQUENCE_LENGTH}):")
    print(f"  > Acc no Teste: {accuracy:.4f}")
    print(f"  > AUC no Teste: {auc:.4f}")
    
except FileNotFoundError:
    print("\nERRO: Certifique-se de que os arquivos .npy (X_donor_encoded.npy, X_acceptor_encoded.npy, y_labels.npy) estão no diretório correto.")
except ValueError as e:
    print(f"\nERRO de valor: {e}. Verifique se o formato dos seus dados de entrada (input shape) está correto.")