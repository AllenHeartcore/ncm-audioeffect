from typing import Union
import warnings

import numpy as np
import zlib


def read_encfile(filename: str) -> tuple:

    toint = lambda x: int.from_bytes(x, byteorder="big")

    with open(filename, "rb") as fin:

        magic = fin.read(4)
        assert magic == b"NCAE"

        ldata = toint(fin.read(4))
        fin.read(8)  # unused

        lkey = toint(fin.read(1)) - 1
        assert lkey % 4 == 0
        key0 = bytearray(fin.read(lkey + 1))
        key = np.array(key0[:4] + key0[5:]) ^ key0[4]

        data = bytearray(fin.read(ldata))

    return key, data


class NCAEDecryptor:

    def __init__(self, key: Union[bytes, bytearray, np.ndarray]):

        table = np.arange(0x100, dtype=np.uint8)
        byte = 0

        for i in range(0x100):
            byte = int(table[i]) + int(key[i % len(key)]) + byte & 0xFF
            table[i], table[byte] = table[byte], table[i]

        self._table = table

    def decrypt(self, data: Union[bytes, bytearray, np.ndarray]) -> bytes:

        for j in range(len(data)):
            byte = int(self._table[(j + 1) & 0xFF])
            data[j] ^= self._table[int(self._table[(byte + j + 1) & 0xFF]) + byte & 0xFF]

        return zlib.decompress(data, -zlib.MAX_WBITS)
