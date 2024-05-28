from openfhe import *

import uuid
import os
import math

import torch
import numpy as np

from flwr.common.logger import log
from logging import INFO

import pickle

MULT_DEPTH = 1
SCALE_MOD_SIZE = 50
BATCH_SIZE = 8
BLOCK_SIZE = 8192

class FheCryptoAPI:
    @staticmethod
    def create_crypto_context_and_keys(
        PKE=True,
        keyswitch=True,
        leveledshe=True,
        mult_depth=MULT_DEPTH,
        scale_mod_sz=SCALE_MOD_SIZE,
        batch_sz=BATCH_SIZE
    ):
        parameters = CCParamsCKKSRNS()
        parameters.SetMultiplicativeDepth(mult_depth)
        parameters.SetScalingModSize(scale_mod_sz)
        parameters.SetBatchSize(batch_sz)
        parameters.SetRingDim(1 << 16)
        
        cc = GenCryptoContext(parameters)
        if PKE:
            cc.Enable(PKESchemeFeature.PKE)
        if keyswitch:
            cc.Enable(PKESchemeFeature.KEYSWITCH)
        if leveledshe:
            cc.Enable(PKESchemeFeature.LEVELEDSHE)

        keys = cc.KeyGen()
        cc.EvalMultKeyGen(keys.secretKey)

        cc_bytes = FheCryptoAPI.serialize_to_bytes(cc)
        pubkey_bytes = FheCryptoAPI.serialize_to_bytes(keys.publicKey)
        seckey_bytes = FheCryptoAPI.serialize_to_bytes(keys.secretKey)

        return cc_bytes, pubkey_bytes, seckey_bytes
    
    @staticmethod
    def __deserialize_bin_file(bytedata: bytes, deserialize_fn):
        filename = f"{uuid.uuid4()}"

        with open(filename, "wb") as f:
            f.write(bytedata)

        obj, res = deserialize_fn(filename, BINARY)

        if not res:
            raise Exception("Unable to deserialize object")
        
        os.remove(filename)
        return obj
    
    @staticmethod
    def serialize_to_bytes(obj):
        filename = f"{uuid.uuid4()}"

        SerializeToFile(filename, obj, BINARY)

        with open(filename, "rb") as f:
            bytedata = f.read()

        os.remove(filename)
        return bytedata

    @staticmethod
    def deserialize_crypto_context(ccbytes: bytes):
        return FheCryptoAPI.__deserialize_bin_file(
            ccbytes,
            DeserializeCryptoContext
        )
    
    @staticmethod
    def deserialize_public_key(keybytes: bytes):
        return FheCryptoAPI.__deserialize_bin_file(
            keybytes,
            DeserializePublicKey
        )
    
    @staticmethod
    def deserialize_private_key(keybytes: bytes):
        return FheCryptoAPI.__deserialize_bin_file(
            keybytes,
            DeserializePrivateKey
        )
    
    @staticmethod
    def deserialize_ciphertext(ciphertext_bytes: bytes):
        return FheCryptoAPI.__deserialize_bin_file(
            ciphertext_bytes,
            DeserializeCiphertext
        )
    
    @staticmethod
    def encrypt_numpy_array(
        cc_bytes: bytes,
        pubkey_bytes: bytes,
        arr: np.array,
        block_sz=BLOCK_SIZE
    ):
        cc = FheCryptoAPI.deserialize_crypto_context(cc_bytes)
        pubkey = FheCryptoAPI.deserialize_public_key(pubkey_bytes)

        enc_arr = []

        arr = arr.flatten()
        for i in range(0, len(arr), block_sz):
            plaintext = cc.MakeCKKSPackedPlaintext(arr[i:i+block_sz], slots=block_sz)
            ciphertext = cc.Encrypt(pubkey, plaintext)
            enc_arr.append(FheCryptoAPI.serialize_to_bytes(ciphertext))

        return enc_arr
    
    @staticmethod
    def decrypt_torch_tensor(
        cc_bytes: bytes,
        seckey_bytes: bytes,
        ciphertext_blocks,
        dtype,
        shape
    ):
        cc = FheCryptoAPI.deserialize_crypto_context(cc_bytes)
        seckey = FheCryptoAPI.deserialize_private_key(seckey_bytes)
        # ciphertext = [FheCryptoAPI.deserialize_ciphertext(ciphertext_blocks)]
        # plaintext = cc.Decrypt(seckey, ciphertext).GetCKKSPackedValue()
        # real_arr = [x.real for x in plaintext]

        decrypted_tensors = []
        for block in ciphertext_blocks:
            cmplx_block = cc.Decrypt(seckey, FheCryptoAPI.deserialize_ciphertext(block)).GetCKKSPackedValue()
            real_tensor = [x.real for x in cmplx_block]
            decrypted_tensors.append(torch.tensor(real_tensor, dtype=dtype))

        flat_sz = math.prod(shape)

        return torch.cat(decrypted_tensors, dim=0)[:flat_sz].reshape(shape).cpu()
    
    @staticmethod
    def secure_fedavg_aggregate_updates(cc_bytes: bytes, updates, fractions):
        cc = FheCryptoAPI.deserialize_crypto_context(cc_bytes)
        aggregated = []

        for upd, f in zip(updates, fractions):
            for i, layer_blocks in enumerate(upd):
                normalized = []
                enc_layer_blocks = pickle.loads(layer_blocks)
                for block in enc_layer_blocks:
                    cipher = FheCryptoAPI.deserialize_ciphertext(block)
                    normalized.append(cc.EvalMult(cipher, f))

                if i >= len(aggregated):
                    aggregated.append(normalized)
                else:
                    for j in range(len(aggregated[i])):
                        aggregated[i][j] = cc.EvalAdd(aggregated[i][j], normalized[j])

        output = []
        for layer in aggregated:
            layer_blocks = []
            for block in layer:
                layer_blocks.append(FheCryptoAPI.serialize_to_bytes(block))
            output.append(pickle.dumps(layer_blocks))

        return output