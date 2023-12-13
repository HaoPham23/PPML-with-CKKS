import tenseal as ts
import numpy as np
from time import time
from utils import *
from flask import Flask, request, jsonify

app = Flask(__name__)
PORT = 5000

n_X = 15
n_H = 3
n_Y = 1

class EncryptedModel:
    def __init__(self):
        loaded_weights = np.load('model_weights.npz')
        # Access individual weights
        self.W1 = loaded_weights['W1'].T.tolist()
        self.b1 = loaded_weights['b1'].squeeze().tolist()
        self.W2 = loaded_weights['W2'].T.tolist()
        self.b2 = loaded_weights['b2'].squeeze().tolist()
        # print(self.W1)
        # print(self.b1)
        # print(self.W2)
        # print(self.b2)
        
    @staticmethod
    def tanh_approx(enc_X: ts.CKKSVector) -> ts.CKKSVector:
        # tanh(x) ~= 0.249476365628036*x - 0.00163574303018748*x^3
        return enc_X.polyval([0, 0.249476365628036, 0, -0.00163574303018748])
    
    def forward_propagation(self, enc_X: ts.CKKSVector) -> ts.CKKSVector:
        enc_Z1 = enc_X.matmul(self.W1) + self.b1
        enc_A1 = EncryptedModel.tanh_approx(enc_Z1)
        enc_Z2 = enc_A1.matmul(self.W2) + self.b2
        # enc_A2 = EncryptedModel.sigmoid_approx(enc_Z2)
        # We dont need to calculate sigmoid, it will be calculate after decryption to avoid error.
        return enc_Z2
    
    def predict(self, enc_X: ts.CKKSVector) -> ts.CKKSVector:
        z2 = self.forward_propagation(enc_X)
        return z2

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("[+] Received a request!")
        start = time()
        data = request.json
        public_key = data.get('public_key')
        print(f"    [-] Received public key, size = {to_megabytes(len(public_key))} MB")
        public_context = ts.context_from(from_base64(public_key))
        enc_input = data.get('encrypted_input')
        print(f"    [-] Received encrypted input, size = {to_megabytes(len(enc_input))} MB")
        print(f"    [-] Trying to predict on encrypted input...")
        enc_input = ts.lazy_ckks_vector_from(from_base64(enc_input))
        enc_input.link_context(public_context)
        enc_output = to_base64(enc_model.predict(enc_input).serialize())
        end = time()
        print(f"    [-] Done, encrypted output size = {to_megabytes(len(enc_output))} MB")
        print(f"[+] Total time: {end - start} s")
        return jsonify({'encrypted_output': enc_output.decode()})

    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    print("-----------Server-----------")
    print("[+] Loading model")
    enc_model = EncryptedModel()
    print("    [-] Model loaded!")
    app.run(port=PORT)
    print(f"[+] Server is running at port {PORT}!")
    