import torch

import sys 
sys.path.append('./')

from model import (ghostnet, FocusedConvLSTM, encoder_info, 
    get_scale, MixDecoder, CenternetHeads, SearchModel)



if __name__ == '__main__':
    encoder = ghostnet(pretrained=True)
    
    input_size = (640, 512)
    reducer = FocusedConvLSTM(1, 32, 3)

    encoder = ghostnet()

    channels = encoder_info(encoder, 3, *input_size)
    scale = get_scale(encoder, 3, *input_size, down_ratio=4)
    decoder = MixDecoder(scale, channels, down_ratio=4, hidden_channels=256, out_channels=64)
    
    heads = {'cpt_hm': 1, 'cpt_off': 2, 'wh': 2}
    classifier = CenternetHeads(heads, in_channels=64)

    model = SearchModel(reducer, encoder, decoder, classifier)
    model.info(13, *input_size, verbose=True)

    # sanity check:
    x = torch.randn(1, 13, 640, 512)
    out = model(x)
    print('\n'.join([str(o.shape) for o in out.values()]), end='\n\n')