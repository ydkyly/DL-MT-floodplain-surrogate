import h5py
import numpy as np
'''
用于读取HEC-RAS自带的HDF5文件中的所有信息
其中最有用的就是流速和水深信息
可以直接输出所有网格水深和流速的时间序列信息，少去把HEC-RAS导出成为tif再转化为npy文件的步骤
'''


import numpy as np
from pathlib import Path

def rmse_between_npy(a_path: str, b_path: str) -> float:
    a_path, b_path = Path(a_path), Path(b_path)

    A = np.load(a_path, allow_pickle=False)
    B = np.load(b_path, allow_pickle=False)

    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")

    diff = A.astype(np.float64) - B.astype(np.float64)
    rmse = np.sqrt(np.mean(diff**2))
    return float(rmse)


def explore_hdf5(group, indent=0):
    # recursively explore HDF5 structure
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            print("    "* indent + f"[Group] {key}")
            explore_hdf5(item, indent + 1) # Recurse into subgroups
        else:
            print("    "* indent + f"[Dataset]{key} -> shape: {item.shape}, dtype:{item.dtype}")

def list_groups(hdf5_group):
    groups = [key for key in hdf5_group.keys() if isinstance(hdf5_group[key], h5py.Group)]
    print('Groups:', groups)

def list_datasets(hdf5_group):
    datasets = [key for key in hdf5_group.keys() if isinstance(hdf5_group[key], h5py.Group)]
    print('Datasets:', datasets)
    # 改成你的两个文件路径

with h5py.File(r"D:\Work\qyb\ResNet-18\data_dw\test\hdf\01.hdf","r") as f:
    # 读取hdf文件的所有内容
    # print("HDF5 File Structure:")
    # explore_hdf5(f)

    # group = f["Results"]['Unsteady']['Output']['Output Blocks']
    # list_groups(group)
    # list_datasets(group)

    # print data in dataset  可以直接读取hecras生产的hdf文件中水深时间序列的信息
    # dataset_path_vx = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Perimeter 1/Cell Velocity - Velocity X'
    # dataset_path_vy = 'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Perimeter 1/Cell Velocity - Velocity Y'
    #
    # dataset_vx = f[dataset_path_vx]
    # data_vx = dataset_vx[:]
    # dataset_vy = f[dataset_path_vy]
    # data_vy = dataset_vy[:]
    # print(f'Extracted data from {dataset_path_vx}:')
    # print(data_vx)
    # print(data_vx[23].min(), data_vx[23].max())
    # print(data_vy[23].min(), data_vy[23].max())
    # print(dataset_vx.shape)
    #
    # val_path_vx = r'D:\Work\qyb\ResNet-18\V2.4\Result_SVD\20251217_084205_1.0_k256_depth, flow_resnet_basic\03_vmag_true.npy'  # 替换为你的文件路径
    # pred_path_vx =r'D:\Work\qyb\ResNet-18\V2.4\Result_SVD\20251217_084205_1.0_k256_depth, flow_resnet_basic\03_vmag_pred.npy'
    #
    # # 读取 .npy 文件
    # data = np.load(val_path_vx)
    # print(data[23].min(), data[23].max())
    #
    # diff = np.abs(data_vx - data)
    # print(diff)
    # print("max abs diff:", diff.max(), "mean abs diff:", diff.mean())

    # # 显示数据内容
    # print("val数据内容：")
    # print(data)

    dataset_path = "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Perimeter 1/Cell Hydraulic Depth"
    dataset = f[dataset_path]
    data2 = dataset[:]
    # print(f'Extracted data from {dataset_path}:')
    print(data2)
    print(data2.min(), data2.max())
    # print(dataset.shape)

    # val_path = r'D:\Work\qyb\ResNet-18\V2.4\Result\20251127_100755_0.5_k256_depth, flow_tcn\06_Y_true.npy'  # 替换为你的文件路径
    # pred_path = r'D:\Work\qyb\ResNet-18\V2.4\Result\20251127_100755_0.5_k256_depth, flow_tcn\06_Y_pred.npy'  # 替换为你的文件路径
    #
    # # 读取 .npy 文件
    # data = np.load(pred_path)
    # print(data.min(), data.max())
    # print(data.round(2))
    #
    # diff = np.abs(data2 - data)
    # # print(diff)
    # print("max abs diff:", diff.max(), "mean abs diff:", diff.mean())


    # rmse = rmse_between_npy(val_path_vx, pred_path_vx)
    # print(f"RMSE = {rmse:.6f}")

