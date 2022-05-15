import torch
import os
import sys
sys.path.append('../LIAN')
sys.path.append('./')

from cvp.models import build_model
from lian.config import predict_argument_parser, predict_setup

def main(args):
    os.environ["EXPORT_ONNX_MODEL"] = "ON"

    cfg = predict_setup(args)
    predictor = build_model(cfg)
    weight = torch.load(args.weight_path, map_location="cpu")['model']
    predictor.load_state_dict(weight)
    predictor = predictor.eval().cuda()
    in_tensor = torch.rand(1, 3, 480, 480).cuda()
    predictor.forward(in_tensor)

    if cfg.dataset.task == "pld":
        onnx_file = "pld.onnx"
        torch.onnx.export(predictor, (in_tensor, ), onnx_file,
                            input_names=['input_0'],
                            output_names=['output_0', 'output_1'])
    else:
        onnx_file = "fs.onnx"
        torch.onnx.export(predictor, (in_tensor, ), onnx_file,
                            input_names=['input_0'],
                            output_names=['output_0'])


if __name__ == "__main__":
    args = predict_argument_parser().parse_args()
    main(args)