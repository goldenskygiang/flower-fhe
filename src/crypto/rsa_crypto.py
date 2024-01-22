import os
import numpy as np
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes, padding
from cryptography.hazmat.primitives.asymmetric import rsa, padding as a_padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

"""
Encryption/Decryption flow:
1. Each client generates their own public/private key pair.
2. Server generates a SYMMETRIC AES key to encrypt/decrypt model.
3. Server distributes the symmetric key to each client,
   by encrypting the key with their public keys.
4. Each client decrypts the symmetric key,
   then use the symmetric key to decrypt the model.
5. Prior sending update, each client encrypts the model
   with the symmetric key with the server's public key,
   and sends the encrypted model to the server.
6. The server decrypts the model to perform aggregation.
"""
class RsaCryptoAPI:
    @staticmethod
    def get_public_key_obj(public_key_pem: bytes):
        return serialization.load_pem_public_key(public_key_pem, backend=default_backend())
    
    @staticmethod
    def get_private_key_obj(private_key_pem: bytes, password=None):
        return serialization.load_pem_private_key(
            private_key_pem,
            password=password,
            backend=default_backend())
    
    @staticmethod
    def check_key_pair_matches(public_key_pem: bytes, private_key_pem: bytes, password=None):
        public_key = RsaCryptoAPI.get_public_key_obj(public_key_pem)
        private_key = RsaCryptoAPI.get_private_key_obj(private_key_pem, password)

        return public_key.public_numbers() == private_key.public_key().public_numbers()
    
    @staticmethod
    def gen_rsa_key_pem(rsa_public_exp=65537, rsa_key_size=2048):
        private_key = rsa.generate_private_key(
            public_exponent=rsa_public_exp,
            key_size=rsa_key_size,
            backend=default_backend()
        )
        public_key = private_key.public_key()

        public_key_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        private_key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        return public_key_pem, private_key_pem
    
    @staticmethod
    def export_public_key(public_key_pem: bytes, filename='key.pub'):
        # public_pem = self.public_key.public_bytes(
        #     encoding=serialization.Encoding.PEM,
        #     format=serialization.PublicFormat.SubjectPublicKeyInfo,
        # )

        with open(filename, 'wb') as f:
            f.write(public_key_pem)

    @staticmethod
    def export_private_key(private_key_pem: bytes, filename='private_key'):
        # private_pem = self.private_key.private_bytes(
        #     encoding=serialization.Encoding.PEM,
        #     format=serialization.PrivateFormat.PKCS8,
        #     encryption_algorithm=serialization.NoEncryption()
        # )

        with open(filename, 'wb') as f:
            f.write(private_key_pem)

    @staticmethod
    def decrypt_aes_key(private_key_pem: bytes, enc_aes_keybytes: bytes):
        private_key = RsaCryptoAPI.get_private_key_obj(private_key_pem)

        aes_key = private_key.decrypt(
            enc_aes_keybytes,
            a_padding.OAEP(
                mgf=a_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        return aes_key
    
    @staticmethod
    def encrypt_numpy_array(aes_key: bytes, arr: np.array):
        arr_bytes = arr.tobytes()
        #shape = arr.shape
        return RsaCryptoAPI.encrypt_bytes(aes_key, arr_bytes)
    
    @staticmethod
    def encrypt_bytes(aes_key: bytes, obj: bytes):
        # Pad the input data to be a multiple of the block size
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(obj) + padder.finalize()

        # Generate a random IV (Initialization Vector)
        iv = os.urandom(16)

        # Create an AES CBC cipher object
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))

        # Encrypt the data
        encryptor = cipher.encryptor()
        enc_state_dict = encryptor.update(padded_data) + encryptor.finalize()

        # Return the IV and encrypted data
        return iv + enc_state_dict
    
    @staticmethod
    def decrypt_obj(aes_key: bytes, enc_obj_bytes: bytes):        
        # Extract the IV from the ciphertext
        iv = enc_obj_bytes[:16]
        ciphertext = enc_obj_bytes[16:]

        # Create an AES CBC cipher object
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))

        # Decrypt the data
        decryptor = cipher.decryptor()
        decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()

        # Unpad the decrypted data
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        dec_state_dict = unpadder.update(decrypted_data) + unpadder.finalize()

        # Return the decrypted data
        return dec_state_dict
    
    @staticmethod
    def decrypt_numpy_array(aes_key: bytes, obj: bytes, dtype):
        plainarr = RsaCryptoAPI.decrypt_obj(aes_key, obj)
        return np.frombuffer(plainarr, dtype)
    
    @staticmethod
    def gen_aes_key(num_bytes=32):
        return os.urandom(num_bytes)
    
    @staticmethod
    def encrypt_aes_key(aes_key: bytes, recipient_public_keybytes: bytes):
        recipient_public_key = RsaCryptoAPI.get_public_key_obj(recipient_public_keybytes)
        
        cipher = recipient_public_key.encrypt(
            aes_key,
            a_padding.OAEP(
                mgf=a_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        return cipher
