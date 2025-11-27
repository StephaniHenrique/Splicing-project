import numpy as np
from tuner import run_tuner, get_callbacks
from model import build_model
from sklearn.model_selection import train_test_split

X_donor = np.load('X_donor_encoded.npy')
X_acceptor = np.load('X_acceptor_encoded.npy')
y = np.load('y_labels.npy')

print(f"Shape dos Dados do Donor: {X_donor.shape}")
print(f"Shape dos Dados do Acceptor: {X_acceptor.shape}")

 
#Split data
X_donor_train, X_donor_val, X_acceptor_train, X_acceptor_val, y_train, y_val = train_test_split(
    X_donor, X_acceptor, y, test_size=0.1, random_state=42
)

tuner, best_hp = run_tuner(
    X_donor_train, X_acceptor_train, y_train,
    X_donor_val, X_acceptor_val, y_val
)

model = build_model(best_hp)

history = model.fit(
    [X_donor_train, X_acceptor_train],
    y_train,
    validation_data=(
        [X_donor_val, X_acceptor_val],
        y_val
    ),
    epochs=100,
    batch_size=best_hp.get("batch_size"),
    callbacks=get_callbacks(),
    verbose=1
)

model.save("final_splice_model.h5")
print("Final model saved")