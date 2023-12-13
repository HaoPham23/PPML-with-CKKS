# Privacy-preserving Machine Learning with Homomorphic Encryption

- [train_and_test.ipynb](./train_and_test.ipynb): used for training model's parameters and testing encryption.
- [app](./app/): Simulates a simple PPML system's workflow. [How to run](#how-to-run) 

## How to run
1. Run the server:
```sh
cd app/server
python3 server.py
```
2. Run the client:
```sh
cd app/client
python3 client.py
```

## An example log:

### Server:
```console
kali@kali-virtual-machine:~/Desktop/do_an_AI/app/server$ python3 server.py 
-----------Server-----------
[+] Loading model
    [-] Model loaded!
 * Running on http://127.0.0.1:5000
[+] Received a request!
    [-] Received public key, size = 67.27 MB
    [-] Received encrypted input, size = 0.44 MB
    [-] Trying to predict on encrypted input...
    [-] Done, encrypted output size = 0.09 MB
[+] Total time: 2.2699689865112305 s
```

### Client:
```console
kali@kali-virtual-machine:~/Desktop/do_an_AI/app/client$ python3 client.py 
-----------Client-----------
[+] Initializing Key...
    [-] A key pair is generated!
    [-] The private key is stored locally! File size = 0.8 MB
    [-] The public key is stored locally! File size = 67.27 MB
    [-] Initialized!
[+] Picking a random example from dataset...
    [-] Got 3007th data, input = [-0.86705605 -0.06823785  0.02090219  1.01174906  0.92572593 -0.17582307
 -0.07702346 -0.67102175 -0.16245742  0.05140079 -0.42437453 -0.91458328
 -0.6023179  -1.15415546  0.        ], output = 0
    [-] Encrypting input...
    [-] Encrypting time = 0.43648409843444824 s
    [-] Posting encrypted input to server...
    [-] Received response! Decrypting the output...
    [-] Result output = False
    [-] Check? True
[+] Finished! Total time: 7.632829189300537 s
```