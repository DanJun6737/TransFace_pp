import numpy as np
import onnx
import torch

##########
import argparse
from typing import Dict, Union
import io
import numpy as np
import torch
from PIL import Image
##########

def _image_to_bytes(x: torch.Tensor, **kwargs) -> io.BytesIO:
    """
    Take an image in [0, 1] and save it as file bytes using PIL.

    Args:
        x: an image tensor in [C, H, W] order, where C is the number of channels,
            and H, W are the height and width.
        kwargs: any keyword arguments that can be passed to PIL's Image.save().

    Returns:
        The file bytes.

    """
    assert x.min() >= 0
    assert x.max() <= 1
    x = (x * 255).byte().permute(1, 2, 0).cpu().numpy()  # Bytes in H, W, C order

    img = Image.fromarray(x)
    byte_array = io.BytesIO()

    img.save(byte_array, **kwargs)
    return byte_array   

def _bytes_to_int32(byte_array: io.BytesIO) -> torch.Tensor:
    """
    Convert a byte array to int32 values.

    Args:
        byte_array: The input byte array.
    Returns:
        The int32 tensor.
    """
    buf = np.frombuffer(byte_array.getvalue(), dtype=np.uint8)
    # The copy operation is required to avoid a warning about non-writable
    # tensors.
    buf = torch.from_numpy(buf.copy()).to(dtype=torch.int32)
    return buf


def convert_onnx(net, path_module, output, opset=11, simplify=False):
    assert isinstance(net, torch.nn.Module)
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)  
    img = img.astype(np.float)  

    
    img = (img / 255.)   
    
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()  
    
    len_file_bytes = _bytes_to_int32(_image_to_bytes(img[0], format="tiff")).shape[0] 
    img_file_bytes = torch.zeros(img.shape[0], len_file_bytes)    
    for index in range(img.shape[0]):
        each_file_bytes = _bytes_to_int32(_image_to_bytes(img[index], format="tiff"))  
        img_file_bytes[index] = each_file_bytes  

    weight = torch.load(path_module)
    net.load_state_dict(weight, strict=True)
    net.eval()
    
    torch.onnx.export(net, img_file_bytes, output, input_names=["data"], keep_initializers_as_inputs=False, verbose=False, opset_version=opset)
    model = onnx.load(output)
    graph = model.graph
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
    if simplify:
        from onnxsim import simplify
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, output)

    
if __name__ == '__main__':
    import os
    import argparse
    from backbones import get_model

    parser = argparse.ArgumentParser(description='ArcFace PyTorch to onnx')
    parser.add_argument('input', type=str, help='input backbone.pth file or path')
    parser.add_argument('--output', type=str, default=None, help='output onnx path')
    parser.add_argument('--network', type=str, default=None, help='backbone network')
    parser.add_argument('--simplify', type=bool, default=False, help='onnx simplify')
    args = parser.parse_args()
    input_file = args.input
    if os.path.isdir(input_file):
        input_file = os.path.join(input_file, "model.pt")
    assert os.path.exists(input_file)
    # model_name = os.path.basename(os.path.dirname(input_file)).lower()
    # params = model_name.split("_")
    # if len(params) >= 3 and params[1] in ('arcface', 'cosface'):
    #     if args.network is None:
    #         args.network = params[2]
    assert args.network is not None
    print(args)
    backbone_onnx = get_model(args.network, dropout=0.0, fp16=False, num_features=512)
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "model.onnx")
    convert_onnx(backbone_onnx, input_file, args.output, simplify=args.simplify)
