import torch
import torch.nn as nn
from .clip_encoder import CLIPVisionTower,CLIPVisionTowerS2
from .glip_encoder import GLIPVisionTower

# from gllava.model.GLIP.vision_merge import VisionTokenMerge
from gllava.model.GLIP.vis_merge.build import build_model

class MergeFeatVisionTower(nn.Module):
    def __init__(self, vision_tower,use_s2, args, delay_load=False,num_of_kvs=4, version='cross_merge'):
        super().__init__()

        self.is_loaded = False
        # vision_pretrained = "GLIP/model_final_fft.pth"
        vision_pretrained = None

        q_dim, context_dim, vision_hidden_size=1024, 256, 1024
        if use_s2:
            self.clip_vision_tower = CLIPVisionTowerS2(vision_tower, args,delay_load=delay_load)
        else:
            self.clip_vision_tower = CLIPVisionTower(vision_tower, args, delay_load=delay_load)

        self.glip_vision_tower = GLIPVisionTower(vision_pretrained, args, delay_load=delay_load)

        # self.load_model()

        q_dim, context_dim, vision_hidden_size=q_dim, context_dim, vision_hidden_size
        
        self.VTMerge=build_model(q_dim, context_dim, vision_hidden_size,num_of_kvs=num_of_kvs,version=version)

    def load_model(self):
        self.clip_vision_tower.load_model()
        self.glip_vision_tower.load_model(self.device)
        self.image_processor = self.clip_vision_tower.image_processor

        self.is_loaded = True
    
    def forward(self, x, x_glip):

        clip_feat = self.clip_vision_tower.forward(x)

        glip_feat = self.glip_vision_tower.forward(x_glip)
        
        merge_feat = self.VTMerge(clip_feat, glip_feat)
        # print(merge_feat)
        return merge_feat

    def forward_features(self, x, x_glip):
        assert  NotImplementedError

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.clip_vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.clip_vision_tower.parameters()).device

    @property
    def config(self):
        assert NotImplementedError
        pass

    @property
    def hidden_size(self):
        return self.clip_vision_tower.hidden_size

    @property
    def num_patches(self):
        return self.clip_vision_tower.num_patches
