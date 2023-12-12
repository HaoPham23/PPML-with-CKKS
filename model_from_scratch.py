import numpy as np
import pandas as pd
import random

random.seed(0)
n_X = 15
n_H = 3
n_Y = 1

def split_train_test(x, y, ratio=0.3):
    idxs = list(range(len(x)))
    random.shuffle(idxs)
    split_idx = int(len(x)*ratio)
    test_idxs, train_idxs = idxs[:split_idx], idxs[split_idx:]
    return x[train_idxs], y[train_idxs], x[test_idxs], y[test_idxs]

def prepare_heart_disease_data():
    data = pd.read_csv("./data/framingham.csv")
    # Drop target columns
    X = data.drop(['TenYearCHD'], axis=1, inplace=False)
    Y = data['TenYearCHD']
    X = X.apply(lambda x: x.fillna(x.mean()),axis=0)
    # Standardize data
    X = (X - X.mean()) / X.std()
    return split_train_test(np.array(X), np.array(Y))

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class ScratchModel:
    def __init__(self):
        self.W1 = np.random.randn(n_H, n_X) * 0.01
        self.b1 = np.zeros((n_H, 1))
        self.W2 = np.random.randn(n_Y, n_H)
        self.b2 = np.zeros((n_Y, 1))
    
    def forward_propagation(self, X):
        Z1 = self.W1 @ X + self.b1
        A1 = np.tanh(Z1)
        Z2 = self.W2 @ A1 + self.b2
        A2 = sigmoid(Z2)
        history = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, history
    
    def cost(self, A2, Y):
        m = Y.shape[1]
        L = np.multiply(Y, np.log(A2)) + np.multiply(1-Y, np.log(1-A2))
        J = (-1/m)*np.sum(L)
        return np.squeeze(J)
    
    def backward_propagation(self, history, X, Y):
        m = Y.shape[1]
        Z1, A1, Z2, A2 = history["Z1"], history["A1"], history["Z2"], history["A2"]
        dZ2 = A2 - Y
        dW2 = 1/m * (dZ2 @ A1.T)
        db2 = 1/m * np.sum(dZ2, axis=-1, keepdims=True)
        dZ1 = (self.W2.T @ dZ2) * (1 - A1**2)
        dW1 = 1/m * (dZ1 @ X.T)
        db1 = 1/m * np.sum(dZ1, axis=-1, keepdims=True)
        
        gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return gradients
        
    def update(self, gradients, learning_rate=0.01):
        self.W1 -= learning_rate * gradients["dW1"]
        self.b1 -= learning_rate * gradients["db1"]
        self.W2 -= learning_rate * gradients["dW2"]
        self.b2 -= learning_rate * gradients["db2"]
        return
        
    def train(self, X, Y, X_test, Y_test, learning_rate, iters):
        for i in range(iters):
            A2, history = self.forward_propagation(X)
            J = self.cost(A2, Y)
            grads = self.backward_propagation(history, X, Y)
            self.update(grads, learning_rate)
            if i % 1000 == 0:
                accuracy = self.test(X_test, Y_test)
                print(f"Cost after iterations {i}: {J} - accuracy: {accuracy}")
    
    def predict(self, x):
        a2, _ = self.forward_propagation(x)
        return a2
    
    def test(self, X, Y) -> float:
        m = Y.shape[1]
        A2, _ = self.forward_propagation(X)
        A2 = A2 > 0.5
        accuracy = np.sum((np.multiply(Y,A2) + np.multiply((1-Y),(1 - A2)))) / float(m) * 100
        return accuracy

import tenseal as ts
class EncryptedModel:
    def __init__(self, plain_model: ScratchModel):
        self.W1 = plain_model.W1.T.tolist()
        self.b1 = plain_model.b1.squeeze().tolist()
        self.W2 = plain_model.W2.T.tolist()
        self.b2 = plain_model.b2.squeeze().tolist()

    # @staticmethod
    # def sigmoid_approx(enc_X: ts.CKKSVector) -> ts.CKKSVector:
    #     # sigmoid(x) ~= 0.5 + 0.197x - 0.004x^3
    #     return enc_X.polyval([0.5, 0.197, 0, -0.004])
    
    @staticmethod
    def tanh_approx(enc_X: ts.CKKSVector) -> ts.CKKSVector:
        # tanh(x) ~= 0.249476365628036*x - 0.00163574303018748*x^3
        return enc_X.polyval([0, 0.249476365628036, 0, -0.00163574303018748])
    
    def forward_propagation(self, enc_X: ts.CKKSVector) -> ts.CKKSVector:
        enc_Z1 = enc_X.matmul(self.W1) + self.b1
        enc_A1 = EncryptedModel.tanh_approx(enc_Z1)
        enc_Z2 = enc_A1.matmul(self.W2) + self.b2
        # enc_A2 = EncryptedModel.sigmoid_approx(enc_Z2)
        return enc_Z2
    
    def predict(self, enc_X: ts.CKKSVector) -> ts.CKKSVector:
        z2 = self.forward_propagation(enc_X)
        return z2

def write_data(file_name: str, data: bytes):
    # data = base64.b64encode(data)
    with open(file_name, 'wb') as f: 
        f.write(data)
  
def read_data(file_name: str) -> bytes:
    with open(file_name, 'rb') as f:
        data = f.read()
    # return base64.b64decode(data)
if __name__ == '__main__':
    x_train, y_train, x_test, y_test = prepare_heart_disease_data()
    y_train = y_train.reshape((1, -1))
    y_test = y_test.reshape((1, -1))
    
    scratch_model = ScratchModel()
    scratch_model.train(x_train.T, y_train, x_test.T, y_test, learning_rate=0.01, iters=4000)
    accuracy = scratch_model.test(x_test.T, y_test)
    print(f"The accuracy is: {accuracy}")

    enc_model = EncryptedModel(scratch_model)
    
    # parameters
    poly_mod_degree = 8192
    bits_scale = 26
    # create TenSEALContext
    ctx_eval = ts.context(
        ts.SCHEME_TYPE.CKKS, 
        poly_mod_degree, -1, 
        coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, 31])
    # scale of ciphertext to use
    ctx_eval.global_scale = 2 ** bits_scale
    # this key is needed for doing dot-product operations
    ctx_eval.generate_galois_keys()
    
    from time import time
    t_start = time()
    enc_x_test = [ts.ckks_vector(ctx_eval, x.tolist()) for x in x_test]
    t_end = time()
    print(f"Encryption of the test-set took {int(t_end - t_start)} seconds")
    
    from tqdm import tqdm
    def encrypted_evaluation(enc_model: EncryptedModel, enc_x_test, y_test):
        t_start = time()
        
        correct = 0
        for enc_x, y in tqdm(zip(enc_x_test, y_test)):
            # encrypted evaluation
            enc_out = enc_model.predict(enc_x)
            # plain comparison
            out = enc_out.decrypt()[0]
            out = sigmoid(out) > 0.5
            if out == y:
                correct += 1
        
        t_end = time()
        print(f"Evaluated test_set of {len(x_test)} entries in {int(t_end - t_start)} seconds")
        print(f"Accuracy: {correct}/{len(x_test)} = {correct / len(x_test)}")
        return correct / len(x_test)
        

    encrypted_accuracy = encrypted_evaluation(enc_model, enc_x_test, y_test.T)
    diff_accuracy = accuracy - encrypted_accuracy
        
    
            