import torch, argparse
import torch.nn as nn
from diffusers import AutoencoderKL

def convert_module(model: AutoencoderKL):
    conv_in = model.encoder.conv_in
    conv_in_new = nn.Conv2d(
        4,
        conv_in.out_channels,
        conv_in.kernel_size,
        conv_in.stride,
        conv_in.padding
    )
    with torch.no_grad():
        conv_in_new.weight[:, :3] = conv_in.weight
        conv_in_new.weight[:, 3:] = 0       
        conv_in_new.bias.copy_(conv_in.bias)
    model.encoder.conv_in = conv_in_new

    conv_out = model.decoder.conv_out
    conv_out_new = nn.Conv2d(
        conv_out.in_channels,
        4,
        conv_out.kernel_size,
        conv_out.stride,
        conv_out.padding
    )
    with torch.no_grad():
        conv_out_new.weight[:3] = conv_out.weight
        conv_out_new.weight[3:] = 0              
        conv_out_new.bias[:3] = conv_out.bias
        conv_out_new.bias[3] = 1
    model.decoder.conv_out = conv_out_new
    
    config = dict(model._internal_dict)
    config.update({
        "in_channels": 4,
        "out_channels": 4,
    })
    model._internal_dict = config

    return model

def main():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--src", type=str, required=True, help="source model path")
    arg_parse.add_argument("--dst", type=str, required=True, help="destination model path")
    args = arg_parse.parse_args()
    
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(args.src)
    converted_vae = convert_module(vae)
    converted_vae.save_pretrained(args.dst)

if __name__ == '__main__':
    main()
