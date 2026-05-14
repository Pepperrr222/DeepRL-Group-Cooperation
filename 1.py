import torch

def check_cuda():
    print(f"PyTorch 版本: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA 已成功启用！")
        print(f"   显卡型号: {torch.cuda.get_device_name(0)}")
        print(f"   显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("❌ CUDA 不可用。")
        print("可能原因:")
        print("1. 未安装 GPU 驱动或版本过低。")
        print("2. 安装的是 CPU 版 PyTorch (检查 --index-url 是否正确)。")
        print("3. Docker 容器内未正确挂载 GPU。")

if __name__ == "__main__":
    check_cuda()