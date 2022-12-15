import lmdb
import json
import pickle
import base64
import clip
import torch
import h5py
# def str2obj(strr):
#     return pickle.loads(base64.b64decode(strr))

# db = lmdb.open('/data/haotian/HAKE-Action-Torch/Data/Trainval_HAKE')
# txn_db = db.begin(write=False)
# SPLITS = [x for x in list(json.load(open('/data/haotian/HAKE-Action-Torch/Data/metadata/data_path.json','r')).keys()) if 'test' not in x]
# image_list = [key for key, _ in txn_db.cursor() if key.decode().split('/')[0] in SPLITS]

# idx = 0
# current_key_raw = image_list[idx]
# image_id          = current_key_raw.decode()
# print(image_id)
# current_data = str2obj(txn_db.get(current_key_raw))
# print(current_data[1])
# a = pickle.load(open('/data/haotian/HAKE-Action-Torch/Data/metadata/gt_pasta_data.pkl','rb'))
# print(a[0][0])

# device = "cuda" if torch.cuda.is_available() else "cpu"
# text_inputs = [clip.tokenize(f"there is no part in the image")]
# out = torch.cat(text_inputs,dim = 0)
# print(out.size())

f = h5py.File("/data/haotian/HAKE-Action-Torch/results/finetune-foot/model_50000.pth_results/collect/HOI_PVP_00029386.jpg.hdf5", "r")    # mode = {'w', 'r', 'a'}

# Print the keys of groups and datasets under '/'.
print(f.filename, ":")
print([key for key in f.keys()], "\n")  
d = f["verb_score"]
print(d[:])