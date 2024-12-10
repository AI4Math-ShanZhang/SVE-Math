datasets = [
    {
        'type': "alignment",
        'ann_file': '/data/dataset/MAVIS/MAVIS_Caption/MAVIS_Caption.json',
        'img_prefix': '/data/dataset/MAVIS/MAVIS_Caption/',
        'conv_temp': 'llava'
    },
    {
        'type': "instruct",
        'ann_file': '/data/dataset/MAVIS/Caption_to_QA/Function_Caption_to_Question.json',
        'img_prefix': '/data/dataset/MAVIS/Caption_to_QA/function_wo/images_wo/',
        'ratio': 1.,
        'conv_temp': 'llava'
    },
    {
        'type': "instruct",
        'ann_file': '/data/dataset/MAVIS/Caption_to_QA/Geometry_Caption_to_Question.json',
        'img_prefix': '/data/dataset/MAVIS/Caption_to_QA/geo_cap_to_question/',
        'ratio': 1.,
        'conv_temp': 'llava'
    },
    {
        'type': "instruct",
        'ann_file': '/data/dataset/MAVIS/DataEngine_Geometry/DataEngine_Geometry.json',
        'img_prefix': '/data/dataset/MAVIS/DataEngine_Geometry/rule_base_geo_vision_dom/',
        'ratio': 1.,
        'conv_temp': 'llava'
    },
    {
        'type': "instruct",
        'ann_file': '/data/dataset/MAVIS/Existing_Dataset_Augment/Existing_Dataset_Augment.json',
        'img_prefix': '/data/dataset/MAVIS/Existing_Dataset_Augment/',
        'ratio': 1.,
        'conv_temp': 'llava'
    },
    {
        'type': "instruct",
        'ann_file': '/data/dataset/MAVIS/Meta_Question/Meta_Question.json',
        'img_prefix': '/data/dataset/MAVIS/Meta_Question/meta_gen/',
        'ratio': 1.,
        'conv_temp': 'llava'
    },
]