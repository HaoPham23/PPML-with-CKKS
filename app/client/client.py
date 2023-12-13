from time import time
import tenseal as ts
from utils import *
import requests
import json

URL = 'http://127.0.0.1:5000/predict'

def initialize_key():
    # parameters
    poly_mod_degree = 8192
    bits_scale = 26
    context = ts.context(
        ts.SCHEME_TYPE.CKKS, 
        poly_mod_degree, -1, 
        coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, 31])
    context.global_scale = 2 ** bits_scale
    # this key is needed for doing dot-product operations
    context.generate_galois_keys()
    print( "    [-] A key pair is generated!")
    # Store Private key
    secret_context = context.serialize(save_secret_key = True)
    secret_size = write_data('secret.txt', secret_context)
    print(f"    [-] The private key is stored locally! File size = {to_megabytes(secret_size)} MB")
    # Drop private key
    context.make_context_public()
    # Store public key
    public_context = context.serialize()
    public_size = write_data('public.txt', public_context)
    print(f"    [-] The public key is stored locally! File size = {to_megabytes(public_size)} MB")
    return

def encrypt(input) -> bytes:
    start = time()
    context = ts.context_from(read_data('secret.txt'))
    enc_input = ts.ckks_vector(context, input.tolist())
    end = time()
    print(f"    [-] Encrypting time = {end-start} s")
    return to_base64(enc_input.serialize())

if __name__=='__main__':
    print("-----------Client-----------")
    start = time()
    print("[+] Initializing Key...")
    initialize_key()
    print("    [-] Initialized!")
    context = ts.context_from(read_data('secret.txt'))
    print("[+] Picking a random example from dataset...")
    idx, input, output = pick_a_random_data_from_test_set()
    print(f"    [-] Got {idx}th data, input = {input}, output = {output}")
    print(f"    [-] Encrypting input...")
    enc_input = encrypt(input)
    input_data = {'public_key': to_base64(read_data('public.txt')).decode(), 'encrypted_input': enc_input.decode()}
    print(f"    [-] Posting encrypted input to server...")
    response = requests.post(URL, json=input_data)
    enc_output = response.json()["encrypted_output"]
    print(f"    [-] Received response! Decrypting the output...")
    enc_output = ts.lazy_ckks_vector_from(from_base64(enc_output.encode()))
    enc_output.link_context(context)
    out = enc_output.decrypt()[0]
    out = sigmoid(out) > 0.5
    print(f"    [-] Result output = {out}")
    print(f"    [-] Check? {output == out}")
    end = time()
    print(f"[+] Finished! Total time: {end - start} s")