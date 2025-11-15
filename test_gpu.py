# test_gpu.py
import torch

print("CUDA доступна:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Нет")
print("Количество GPU:", torch.cuda.device_count())

if torch.cuda.is_available():
    x = torch.rand(3, 3).cuda()
    print("Тензор на GPU:", x)
else:
    print("GPU не найдена — используем CPU")