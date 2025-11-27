
import keras_tuner as kt
import tensorflow as tf
from model import build_model

def get_callbacks():  
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=10,
            mode="max",
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="best_model.h5",
            monitor="val_auc",
            save_best_only=True,
            save_weights_only=False,
            mode="max"
        )
    ]

def run_tuner(X_donor_train, X_acceptor_train, y_train,
               X_donor_val, X_acceptor_val, y_val):

    tuner = kt.BayesianOptimization(
        build_model,
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=50,
        directory="tuner_results",
        project_name="splice_model"
    )

    tuner.search(
        [X_donor_train, X_acceptor_train],
        y_train,
        validation_data=(
            [X_donor_val, X_acceptor_val],
            y_val
        ),
        epochs=50,
        batch_size=64,
        callbacks=get_callbacks(),
        verbose=1
    )

    #Getting better hyperparameters found
    best_hp = tuner.get_best_hyperparameters(1)[0]
    print("Best hyperparametes:")
    print(best_hp.values)

    return tuner, best_hp