

def build_model(q_dim, context_dim, vision_hidden_size,num_of_kvs,version,**args):
    if version=='identify_sequence' or version=='identify_channel':
        from gllava.model.GLIP.vis_merge.identity import ORi_VisionToken
        model=ORi_VisionToken(q_dim, context_dim, vision_hidden_size,num_of_kvs,version,**args)
    else:
        raise ValueError(f"not support the meger method : '{version}'")
    return model