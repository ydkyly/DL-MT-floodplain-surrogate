import numpy as np


A = np.load(r'D:\Work\qyb\ResNet-18\V2.4\Result_SVD\20251217_084205_1.0_k256_depth, flow_resnet_basic\PostEval\01_Y_pred.npy')
B = np.load(r'D:\Work\qyb\ResNet-18\V2.4\Infer\01_Y_pred.npy' )
print("shape A/B:", A.shape, B.shape)
diff = np.abs(A-B)
print("max abs diff:", diff.max(), "mean abs diff:", diff.mean())

# 指定 .npy 文件路径
# val_path = r''  # 替换为你的文件路径
infer_path = r'D:\Work\qyb\ResNet-18\data_dw\Infer_SVD_resnet\01_Y_pred.npy'  # 替换为你的文件路径

# 读取 .npy 文件
# data = np.load(val_path)

# # 显示数据内容
# print("val数据内容：")
# print(data.min(), data.max())


# 读取 .npy 文件
infer_data = np.load(infer_path)

# 显示数据内容
print("infer数据内容：")
print(infer_data)
print(infer_data.min(), infer_data.max())

# 显示数据类型和形状
# print("\n数据类型：", type(infer_data))
# print("数据形状：", infer_data.shape)
#
# assert data.shape == infer_data.shape
# diff = np.subtract(data, infer_data, dtype=np.float32)  # 避免整型下溢/溢出
# print("验证与推理预测的值差异：")
# print(diff)
# print(np.max(np.abs(diff)))