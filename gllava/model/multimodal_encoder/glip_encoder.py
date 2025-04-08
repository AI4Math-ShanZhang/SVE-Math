import torch
import torch.nn as nn

from gllava.model.GLIP.backbone import swint
from gllava.model.GLIP.backbone import fpn as fpn_module
from gllava.model.GLIP.backbone import BJunc 
from gllava.model.GLIP.backbone.dropblock import DropBlock2D 
from collections import OrderedDict


def load_checkpoint(model, checkpoint):
    state_dict = checkpoint['model']
    new_state_dict_body = {}
    new_state_dict_fpn = {}
    for name, param in state_dict.items():
        if 'body.' in name and "language_backbone" not in name:
            new_name = name.replace('module.backbone.body.', 'body.')
            new_state_dict_body[new_name] = param
        elif 'fpn.' in name:
            new_name = name.replace('module.backbone.fpn.', 'fpn.')
            new_state_dict_fpn[new_name] = param
        elif 'boundary_heads.' in name:
            new_name = name.replace('module.boundary_heads.', 'bjhead.')
            new_state_dict_fpn[new_name] = param
        else:
            continue
    new_state_dict_body.update(new_state_dict_fpn)
    # Load the new state dict into the model
    model.load_state_dict(new_state_dict_body, strict=False)

    print("Visual Encoder Checkpoint loaded successfully.")

def conv_with_kaiming_uniform(use_gn=False, use_relu=False):
    def make_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        conv = torch.nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=dilation * (kernel_size - 1) // 2, 
            dilation=dilation, 
            bias=False if use_gn else True
        )
        # Caffe2 implementation uses XavierFill, which in fact
        # corresponds to kaiming_uniform_ in PyTorch
        nn.init.kaiming_uniform_(conv.weight, a=1)
        if not use_gn:
            nn.init.constant_(conv.bias, 0)
        module = [conv,]
        if use_relu:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv

class GLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        body = swint.build_swint_backbone(frozen_stage=4)
        in_channels_stages = (96, 192, 384, 768)
        out_channels = 256
        in_channels_p6p7 = out_channels
        fpn = fpn_module.FPN(
            in_channels_list=[
                0,
                in_channels_stages[-3],
                in_channels_stages[-2],
                in_channels_stages[-1],
                ],
            out_channels=out_channels,
            conv_block=conv_with_kaiming_uniform(),
            top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
            drop_block=DropBlock2D(0.3, 3),
            use_spp=False,
            use_pan=False,
            return_swint_feature_before_fusion=False
        )
        boundary_heads=BJunc.HourglassNet3D()

        self.glip_model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn),("bjhead", boundary_heads)]))
        if not delay_load:
            self.load_model()

    def load_model(self, device):
        checkpoint = self.vision_tower_name
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                checkpoint =torch.load(f, map_location=device)
            load_checkpoint(self.glip_model,checkpoint)
            
        self.glip_model.eval()
        self.glip_model.requires_grad_(False)

        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):
        with torch.no_grad():
            image_embeddings=self.glip_model(images)
            torch.cuda.empty_cache()
        return image_embeddings

    @property
    def device(self):
        return self.glip_model.device
