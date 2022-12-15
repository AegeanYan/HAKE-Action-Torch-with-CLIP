##################################################################################
#  Author: Hongwei Fan                                                           #
#  E-mail: hwnorm@outlook.com                                                    #
#  Homepage: https://github.com/hwfan                                            #
#  Based on PaStaNet in CVPR'20                                                  #
#  TF version:                                                                   #
#  https://github.com/DirtyHarryLYL/HAKE-Action/tree/Instance-level-HAKE-Action  #
##################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import clip
import torch
from PIL import Image , ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class pasta_res50(nn.Module):

    def __init__(self, cfg):
        super(pasta_res50, self).__init__()
        
        self.cfg             = cfg
        self.num_verbs       = cfg.DATA.NUM_VERBS
        self.num_parts       = cfg.DATA.NUM_PARTS
        self.pasta_idx2name  = cfg.DATA.PASTA_NAMES

        self.pasta_name2idx  = dict()
        self.num_pastas      = []
        for pasta_idx, part_name in enumerate(self.pasta_idx2name):
            self.pasta_name2idx[part_name] = pasta_idx
            self.num_pastas.append(cfg.DATA.NUM_PASTAS[part_name.upper()])
        
        self.num_fc          = cfg.MODEL.NUM_FC
        self.scene_dim       = 1024
        self.human_dim       = 2048
        self.roi_dim         = 1024
        self.part_agg_rule   = cfg.MODEL.PART_AGG_RULE
        self.part_agg_num    = [len(part_agg_rule) for part_agg_rule in self.part_agg_rule]
        if self.cfg.MODEL.PART_ROI_ENABLE:
            self.num_fc_parts  = [part_agg_num*self.roi_dim + self.scene_dim + self.human_dim for part_agg_num in self.part_agg_num]
        else:
            self.num_fc_parts  = [self.scene_dim + self.human_dim for part_agg_num in self.part_agg_num]
        
        if self.cfg.MODEL.POSE_MAP:
            self.num_fc_parts  = [(x + cfg.MODEL.SKELETON_DIM) for x in self.num_fc_parts]
            
        self.module_trained = cfg.MODEL.MODULE_TRAINED
        self.dropout_rate   = cfg.MODEL.DROPOUT
        self.pasta_language_matrix  = torch.from_numpy(np.load(cfg.DATA.PASTA_LANGUAGE_MATRIX_PATH)).cuda()

        self.clip_dim = 512

        ########################
        #     CLIP network     #
        ########################
        # The person's [classes] is [doing].
        self.classes = ['head', 'arm', 'hand', 'hip', 'leg', 'foot']
        self.classes.reverse()   # conform to config.py
        self.characteristics = {
        'head' :  {'eating', 'inspecting', 'talking with something', 'talking to something', 'closing with something', 'kissing', 'put somthing over', 'licking', 'blowing', 'drinking with something', 'smelling', 'wearing', 'listening', 'doing nothing'}, 
        'arm' : {'carrying something', 'close to something', 'hugging', 'swinging', 'crawling', 'dancing', 'playing martial art', 'doing nothing'},
       'hand' : {'holding something', 'carrying something', 'reaching for something', 'touching', 'putting on something', 'twisting', 'wearing something', 'throwing something', 'throwing out something', 'writting on something', 'pointing with something', 'pointing to something', 'using something to point to something', 'pressing something', 'squeezing something', 'scratching something', 'pinching something', 'gesturing to something', 'pushing something', 'pulling something', 'pulling with something', 'washing something', 'washing with something',
                    'holding something in both hands', 'lifting something', 'raising something', 'feeding', 'cutting with something', 'catching with something', 'pouring something into something', 'crawling ', 'dancing', 'playing martial art', 'doing nothing'},
        'hip' : {'sitting on something', 'sitting in something', 'sitting beside something', 'close with something', 'bending', 'doing nothing'}, 
        'leg' : {'walking with something', 'walking to something', 'running with something', 'running to something', 'jumping with something', 'close with something', 'straddling something', 'jumping down', 'walking away', 'bending', 'kneeling', 'crawling', 'dancing', 'playing martial art', 'doing nothing'},
        'foot' : {'standing on something', 'treading on something', 'walking with something', 'walking to something', 'running with something', 'running to something', 'dribbling', 'kicking something', 'jumping down', 'jumping with something', 'walking away', 'crawling', 'dancing', 'falling down', 'playing martial art', 'doing nothing'}}

        # Load the model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load('ViT-B/32', device)
        self.model

        # CLIP feature to PaSta feature.

        self.clip2pasta = nn.ModuleList(
                                            [
                                                nn.Sequential(
                                                    nn.Linear(self.clip_dim, self.num_fc),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(self.num_fc, self.num_fc),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(self.num_fc, self.num_pastas[pasta_idx])
                                                )
                                                for pasta_idx in range(len(self.pasta_idx2name))
                                            ]
                                        )
        
        # Verb classifier.
        
        self.verb_cls_scores = nn.Sequential(
                        nn.Linear(len(self.pasta_idx2name) * self.num_fc, self.num_fc),
                        nn.ReLU(inplace=True),
                        nn.Dropout(self.dropout_rate),
                        nn.Linear(self.num_fc, self.num_verbs)
                    ) 

        ##############################
        # Freeze the useless params. #
        ##############################
            
        for pasta_idx in range(len(self.pasta_idx2name)):
            for p in self.clip2pasta[pasta_idx].parameters():
                p.requires_grad = self.pasta_idx2name[pasta_idx] in self.module_trained

        for p in self.verb_cls_scores.parameters():
            p.requires_grad = 'verb' in self.module_trained


    # image/frame --> resnet --> part RoI features + pose map feature --> PaSta (Part States) recognition --> verb (whole body action) recognition
    def forward(self, image, annos):
            
        f_parts = []
        s_parts = []
        p_parts = []

        # f_part_fc7: 1 x num_fc
        # s_part: 1 x num_pastas
        # p_part = sigmoid(s_part)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        for class_id, classes in enumerate(self.classes):

            try:
                ims = Image.open(image[0])
            except Exception as e:
                print(image[0],e)

            ################################
            #    Image and Text Feature    #
            ################################

            # image_input = self.preprocess(Image.open(image[0])).unsqueeze(0).to(device)
            # text_inputs = [clip.tokenize(f"there is no {classes} in the image")]
            # for characteristics in self.characteristics[classes]:
            #     text_inputs.append(clip.tokenize(f"the person's {classes} is {characteristics}"))
            # text_inputs = torch.cat(text_inputs,dim=0).to(device)

            # with torch.no_grad():
            #     image_features = self.model.encode_image(image_input).float()
            #     text_features = self.model.encode_text(text_inputs).float()

            # image_features = self.clip2pasta[class_id](image_features)

            # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # similarity = (100.0 * image_features @ text_features.T)

            # now_num_pasta = (len(similarity[0]) - 1)
            # s_part = torch.zeros(now_num_pasta)
            # for idx in range(now_num_pasta):
            #     if idx == 0:
            #         s_part[now_num_pasta - 1] = (similarity[0][0] + similarity[0][now_num_pasta]) / 2
            #     else:
            #         s_part[idx - 1] = similarity[0][idx]
            # s_part = s_part - torch.mean(s_part)
            # p_part  = torch.sigmoid(s_part)

            ################################

            ################################
            #      Only Image Feature      #
            ################################

            image_input = self.preprocess(Image.open(image[0])).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_input).float().to(device)

            s_part = self.clip2pasta[class_id](image_features)[0]
            p_part = torch.sigmoid(s_part)

            ################################

            f_parts.append(image_features)
            s_parts.append(s_part.to(device))
            p_parts.append(p_part)
        
        f_pasta_visual = torch.cat(f_parts, 1)
        p_pasta = torch.cat(p_parts, 0)

        # s_verb: 1 x num_verbs
        # print(f_pasta_visual.size()) 1*3072
        # print(p_pasta.size()) 93
        # print(len(self.pasta_idx2name) * self.num_fc) 3072

        # classify the verbs
        s_verb = self.verb_cls_scores(f_pasta_visual)
        p_verb = torch.sigmoid(s_verb)

        f_pasta_language = torch.matmul(p_pasta.to(device), self.pasta_language_matrix)

        # print(f_pasta_language.size()) 1536
        f_pasta_language = f_pasta_language.view(1,-1)
        f_pasta = torch.cat([f_pasta_visual, f_pasta_language], 1)
        
        # return the pasta feature and pasta probs if in test/inference mode, 
        # else return the pasta scores for loss input.
        #import ipdb; ipdb.set_trace()

        if not self.training:
            return f_pasta, p_pasta.to(device), p_verb
        else:
            return s_parts, s_verb.to(device)
        
