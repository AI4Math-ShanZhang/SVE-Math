#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from gllava.model.language_model.llava_llama import LlavaLlamaForCausalLM,LlavaLlamaModel
from gllava.model.GLIP.backbone import swint
from gllava.model.GLIP.backbone import fpn as fpn_module
from gllava.model.GLIP.backbone import BJunc 
from gllava.model.GLIP.backbone.dropblock import DropBlock2D 
from gllava.model.GLIP.vision_merge import VisionTokenMerge
from collections import OrderedDict
import torch
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
class HLlavaMetaModel:
    def __init__(self,config: LlamaConfig,**kwargs):
        super(HLlavaMetaModel, self).__init__(config)
        self.config = config
        # if not hasattr(self.config, "train_encoder"):
        self.vision_pretrained = kwargs.get("vision_pretrained", None)
        self.q_dim, self.context_dim, self.vision_hidden_size=1024,256,1024
        # self.initialize_siwn_modules(self.vision_pretrained)
        # self.projection_layers(self.q_dim, self.context_dim, self.vision_hidden_size)
    def initialize_siwn_modules(self):
        checkpoint=self.vision_pretrained
        body= swint.build_swint_backbone(frozen_stage=4)
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
        model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn),("bjhead", boundary_heads)]))
        self.backbone= model 
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                checkpoint =torch.load(f)#map_location=torch.device("cpu")
            load_checkpoint(model,checkpoint)
        # return model
        self.backbone.body.eval()
        for p in self.backbone.body.parameters():
            p.requires_grad = False
        self.backbone.fpn.eval()
        for p in self.backbone.fpn.parameters():
            p.requires_grad = False
        self.backbone.bjhead.eval()
        for p in self.backbone.bjhead.parameters():
            p.requires_grad = False
        # if config.train_mask_decoder:
        #     self.visual_model.mask_decoder.train()
        #     for param in self.visual_model.mask_decoder.parameters():
        #         param.requires_grad = True
    def initialize_projection_layers(self,num_of_kvs):
        q_dim, context_dim, vision_hidden_size=self.q_dim, self.context_dim, self.vision_hidden_size
        self.VTMerge=VisionTokenMerge(q_dim, context_dim, vision_hidden_size,num_of_kvs=num_of_kvs)
        for param in self.VTMerge.layers.parameters():
            param.requires_grad = True
        # Projection layer
        # in_dim = config.hidden_size#5120
        # out_dim = config.out_dim#256
        # text_fc = [
        #     nn.Linear(in_dim, in_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_dim, out_dim),
        #     nn.Dropout(0.0),
        # ]
        # self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        # self.text_hidden_fcs.train()
        # for param in self.text_hidden_fcs.parameters():
        #     param.requires_grad = True

class LlavaConfig(LlamaConfig):
    model_type = "llava"


class HLlavaLlamaModel(HLlavaMetaModel, LlavaLlamaModel):

    def __init__(self, config: LlamaConfig,**kwargs):
        super(HLlavaLlamaModel, self).__init__(config,**kwargs)


class HLlavaLlamaForCausalLM(LlavaLlamaForCausalLM):

    def __init__(self, config,**kwargs):
        super(HLlavaLlamaForCausalLM, self).__init__(config)
        self.model = HLlavaLlamaModel(config,**kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)
    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings=self.model.backbone(pixel_values)
            torch.cuda.empty_cache()
        return image_embeddings
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    def model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_glips: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        image_glip_features = self.get_visual_embs(image_glips)
        image_clip_features = self.get_model().get_vision_tower()(images)
        
        image_features=self.model.VTMerge(image_clip_features,image_glip_features)
        image_features = self.get_model().mm_projector(image_features)
        output = super().forward(
                images=images,
                image_features=image_features,
                attention_mask=attention_mask,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
        

        return output

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

# AutoConfig.register("llava", LlavaConfig)
# AutoModelForCausalLM.register(LlavaConfig, HLlavaLlamaForCausalLM)
