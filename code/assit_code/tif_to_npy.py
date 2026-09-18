import torch

ckpt = torch.load(r"D:\Work\qyb\ResNet-18\V2.4\Result_SVD\20251217_084205_1.0_k256_depth, flow_resnet_basic\best_20251218_154205_1.0_k256_depth, flow_resnet_basic.pth", map_location="cpu")
# 如果用了 DataParallel
if any(k.startswith("module.") for k in ckpt.keys()):
    ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

keys = list(ckpt.keys())
print("num keys:", len(keys))

# 常见的基底/均值命名
cand = [k for k in keys if ("lowrank" in k.lower()) or k.endswith(".B") or k.endswith(".mu") or k.endswith(".mean")]
for k in cand[:200]:
    v = ckpt[k]
    shape = tuple(v.shape) if hasattr(v, "shape") else type(v)
    print(k, shape)

print("\nHas lowrank.B:", "lowrank.B" in ckpt)
print("Has lowrank.mu:", "lowrank.mu" in ckpt)
