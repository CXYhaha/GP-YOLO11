import warnings
import os
import sys

# 确保使用本地修改版的 ultralytics
sys.path.insert(0, "/home/yangpengchao/caoxinyao/ultralytics-main")

from ultralytics import YOLO

warnings.filterwarnings('ignore')

# ====================== 配置区 ======================
MODEL_CFG = '/home/yangpengchao/caoxinyao/ultralytics-main/ultralytics/cfg/models/11/yolo11.yaml'   # 模型配置文件
PRETRAINED_WEIGHTS = '/home/yangpengchao/caoxinyao/ultralytics-main/yolo11m.pt'  # 预训练权重
DATA_YAML = '/home/yangpengchao/caoxinyao/ultralytics-main/data.yaml'            # 数据集配置
PROJECT_DIR = '/home/yangpengchao/caoxinyao/ultralytics-main/runs'               # 输出目录
EXPERIMENT_NAME = 'cvs_yolo11_multiGPU'                                          # 实验名称

EPOCHS = 300
IMAGE_SIZE = 512         # 输入图像大小
BATCH_SIZE = 32          # 总 Batch Size (4张卡，每张卡等效 8)
WORKERS = 16             # 数据加载线程数
DEVICE = '0,1,2,3'       # 指定使用的 GPU ID
OPTIMIZER = 'SGD'        # 优化器
AMP = True               # 自动混合精度 (节省显存)
CACHE = False            # 缓存图像到内存 (显存够大可开 True 加速)
# ====================================================


def main():
    # ================= 关键修复：强制单进程多卡模式 =================
    # 设置 RANK=-1 和 WORLD_SIZE=1 可以阻止 Ultralytics 启动 
    # torch.distributed.run 子进程，从而避免 "CalledProcessError" 错误。
    # 在这种模式下，Ultralytics 仍会自动利用所有指定的 DEVICE 进行训练。
    os.environ['RANK'] = '-1'
    os.environ['WORLD_SIZE'] = '1'
    # ===============================================================

    print("🚀 启动 YOLOv11 多 GPU 训练任务 ...")
    print(f"使用 GPU: {DEVICE}")
    print(f"模型配置: {MODEL_CFG}")
    print(f"当前生效模式: 单进程多卡 (RANK={os.environ.get('RANK')})")

    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"❌ data.yaml 未找到: {DATA_YAML}")
    
    os.makedirs(PROJECT_DIR, exist_ok=True)

    # 初始化模型
    # 注意：这里传入的是 .yaml 配置文件，表示从头构建架构
    model = YOLO(MODEL_CFG)

    # 加载预训练权重
    if PRETRAINED_WEIGHTS and os.path.exists(PRETRAINED_WEIGHTS):
        print(f"✅ 加载预训练权重: {PRETRAINED_WEIGHTS}")
        model.load(PRETRAINED_WEIGHTS)
    else:
        print("⚠️ 未加载预训练权重，将从头开始训练。")

    # ⚙️ 开始训练
    try:
        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            workers=WORKERS,
            device=DEVICE,
            optimizer=OPTIMIZER,
            amp=AMP,
            cache=CACHE,
            project=PROJECT_DIR,
            name=EXPERIMENT_NAME,
            exist_ok=True,
            
            # --- 关键训练策略 ---
            rect=False,       # 关闭矩形训练 (因为你的图像是正方形 512x512)
            cos_lr=True,      # 开启余弦退火学习率调度
            patience=30,      # 早停耐心值 (30个epoch无提升则停止)
            verbose=True,     # 显示详细日志
            save=True,        # 保存检查点
            save_period=10,   # 每10个epoch保存一次权重 (可选)
        )

        print("\n✅ 多 GPU 训练完成！最优模型保存在：")
        print(f"   {PROJECT_DIR}/{EXPERIMENT_NAME}/weights/best.pt")
        
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        raise e


if __name__ == "__main__":
    main()