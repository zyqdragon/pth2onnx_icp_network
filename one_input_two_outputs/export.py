import torch
import os
import sys
sys.path.append('../upper_directory')

import new_model
from func_file import predict_argument_parser, func_setup

def main(args):
    os.environ["EXPORT_ONNX_MODEL"] = "ON"
    # cft denote configure file
    cft = func_setup(args) 
    model_cfg = new_model(cft)
    weight = torch.load(weight_path, map_location="cpu")['model']
    model_cfg.load_state_dict(weight)
    model_cfg = model_cfg.eval().cuda()
    in_tensor = torch.rand(1, 3, 480, 480).cuda()
    model_cfg.forward(in_tensor)

    onnx_file = "test.onnx"
    torch.onnx.export(model_cfg, (in_tensor, ), onnx_file,
                            input_names=['input_0'],
                            output_names=['output_0', 'output_1']

if __name__ == "__main__":
    args = predict_argument_parser().parse_args()
    main(args)
