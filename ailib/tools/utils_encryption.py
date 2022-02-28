import base64
import urllib
import hashlib

def url_to_base64(url):
    with urllib.request.urlopen(url) as f:
        data = base64.b64encode(f.read())
    return data.decode()

def path_to_base64(path):
    with open(path, "rb") as f:
        content = f.read()
        data = base64.b64encode(content)
    return data.decode()

def to_base64(text):
    if text.startswith("http") or text.startswith("ftp"):
        return url_to_base64(text)
    else:
        return path_to_base64(text)

def md5(text):
    return hashlib.md5(str(text).encode('utf-8')).hexdigest()

def sha256(text):
    return hashlib.sha256(str(text).encode('utf-8')).hexdigest()