import lmdb
import json
import pickle
import base64
def str2obj(strr):
    return pickle.loads(base64.b64decode(strr))

db = lmdb.open('/data/haotian/HAKE-Action-Torch/Data/Trainval_HAKE')
txn_db = db.begin(write=False)
SPLITS = [x for x in list(json.load(open('/data/haotian/HAKE-Action-Torch/Data/metadata/data_path.json','r')).keys()) if 'test' not in x]
image_list = [key for key, _ in txn_db.cursor() if key.decode().split('/')[0] in SPLITS]

idx = 0
current_key_raw = image_list[idx]
image_id          = current_key_raw.decode()
print(image_id)
current_data = str2obj(txn_db.get(current_key_raw))
print(current_data[4])
