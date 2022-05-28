本项目文件阐述如何将pytorch的pth模型转换为onnx模型的例子:
===========

包含单输入多输出模型的转换以及多数入单输出的模型转换。

pth模型的载入和保存代码需要匹配，否则会报如下错误：
--------
torch.nn.modules.module.ModuleAttributeError: ‘Network‘ object has no attribute ‘copy‘

解决办法
匹配即可，例如

case1
若save代码为
torch.save(network.cpu().state_dict(), model_name)
则load的代码应为
network.load_state_dict(torch.load(model_name))

case2
若save代码为
torch.save(network, model_name)
则load的代码应为
network.load_state_dict(torch.load(model_name).cpu().state_dict())
