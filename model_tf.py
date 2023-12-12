
import numpy as np
import pandas as pd
import random

# random.seed(0)

def split_train_test(x, y, ratio=0.3):
    idxs = list(range(len(x)))
    random.shuffle(idxs)
    split_idx = int(len(x)*ratio)
    test_idxs, train_idxs = idxs[:split_idx], idxs[split_idx:]
    return x[train_idxs], y[train_idxs], x[test_idxs], y[test_idxs]

def prepare_heart_disease_data():
    data = pd.read_csv("../data/framingham.csv")
    # Drop target columns
    X = data.drop(['TenYearCHD'], axis=1, inplace=False)
    Y = data['TenYearCHD']
    X = X.apply(lambda x: x.fillna(x.mean()),axis=0)
    # Standardize data
    X = (X - X.mean()) / X.std()
    return split_train_test(np.array(X), np.array(Y))


if __name__ == '__main__':
    import tensorflow as tf
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(15,)),
        tf.keras.layers.Dense(3, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

    x_train, y_train, x_test, y_test = prepare_heart_disease_data()

    print("############# Data summary #############")
    print(f"x_train has shape: {x_train.shape}")
    print(f"y_train has shape: {y_train.shape}")
    print(f"x_test has shape: {x_test.shape}")
    print(f"y_test has shape: {y_test.shape}")
    print("#######################################")

    history = model.fit(x_train, y_train, epochs=10)
    model.evaluate(x_test, y_test)
    # Epoch 1/5
    # 93/93 [==============================] - 1s 3ms/step - loss: 0.8012 - accuracy: 0.1635
    # Epoch 2/5
    # 93/93 [==============================] - 0s 3ms/step - loss: 0.7035 - accuracy: 0.4692
    # Epoch 3/5
    # 93/93 [==============================] - 0s 3ms/step - loss: 0.6277 - accuracy: 0.8028
    # Epoch 4/5
    # 93/93 [==============================] - 0s 3ms/step - loss: 0.5693 - accuracy: 0.8517
    # Epoch 5/5
    # 93/93 [==============================] - 0s 3ms/step - loss: 0.5248 - accuracy: 0.8544
    # 40/40 [==============================] - 0s 3ms/step - loss: 0.5225 - accuracy: 0.8324
    weights = model.get_weights()
    W1, b1, W2, b2 = weights
    print(W1)