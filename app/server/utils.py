import base64

def write_data(file_name: str, data: bytes) -> int:
    data = base64.b64encode(data)
    with open(file_name, 'wb') as f: 
        f.write(data)
    return len(data)

def read_data(file_name: str) -> bytes:
    with open(file_name, 'rb') as f:
        data = f.read()
    return base64.b64decode(data)

def to_base64(data: bytes) -> bytes:
    return base64.b64encode(data)

def from_base64(data: bytes) -> bytes:
    return base64.b64decode(data)

def to_megabytes(size_in_bytes: int) -> float:
    return round(size_in_bytes/2**20, 2)