"""
Main CLI for Brain MRI Segmentation Project

Usage:
    python main.py train    - Train the model
    python main.py infer    - Run inference on a single patient
    python main.py demo     - Run a quick demo with 1 epoch
"""
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

import config
from models import AttentionUNet
from utils import BraTSDataset, mc_dropout_inference, plot_results_with_uncertainty
from train import Trainer


def get_patient_ids(data_dir: str) -> list:
    """
    獲取資料集中的病人 ID 列表
    
    Args:
        data_dir: 資料集根目錄
        
    Returns:
        病人 ID 列表
    """
    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory not found at {data_dir}")
        print(f"Please run: python scripts/download_brats.py")
        return []
    
    patient_ids = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if not patient_ids:
        print(f"❌ Error: No patient folders found in {data_dir}")
        return []
    
    return sorted(patient_ids)


def train_command(args):
    """
    訓練命令
    """
    print("\n" + "="*60)
    print("🚀 Brain MRI Segmentation - Training Mode")
    print("="*60 + "\n")
    
    # 設定隨機種子
    config.set_seed()
    
    # 載入資料
    patient_ids = get_patient_ids(config.DATA_DIR)
    if not patient_ids:
        return
    
    print(f"✓ Found {len(patient_ids)} patients")
    
    # 劃分訓練集和驗證集
    split_idx = int(len(patient_ids) * config.TRAIN_VAL_SPLIT)
    train_ids = patient_ids[:split_idx]
    val_ids = patient_ids[split_idx:]
    
    print(f"✓ Train: {len(train_ids)} patients, Val: {len(val_ids)} patients")
    
    # 建立 Dataset
    train_dataset = BraTSDataset(
        data_dir=config.DATA_DIR,
        patient_ids=train_ids,
        image_size=config.IMAGE_SIZE,
        mode='train',
        use_smart_slice=True
    )
    
    val_dataset = BraTSDataset(
        data_dir=config.DATA_DIR,
        patient_ids=val_ids,
        image_size=config.IMAGE_SIZE,
        mode='val',
        use_smart_slice=True
    )
    
    # 建立 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"✓ DataLoader created")
    
    # 建立模型
    model = AttentionUNet(
        n_channels=config.N_CHANNELS,
        n_classes=config.N_CLASSES,
        dropout_p=config.DROPOUT_P
    ).to(config.DEVICE)
    
    print(f"✓ Model initialized on {config.DEVICE}")
    
    # 建立訓練器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.DEVICE,
        use_amp=torch.cuda.is_available()
    )
    
    # 開始訓練
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    trainer.train(epochs=config.EPOCHS)
    
    print("\n✅ Training completed!")


def infer_command(args):
    """
    推論命令
    """
    print("\n" + "="*60)
    print("🔍 Brain MRI Segmentation - Inference Mode")
    print("="*60 + "\n")
    
    # 檢查模型是否存在
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"❌ Error: Model not found at {config.MODEL_SAVE_PATH}")
        print("Please train the model first: python main.py train")
        return
    
    # 載入資料
    patient_ids = get_patient_ids(config.DATA_DIR)
    if not patient_ids:
        return
    
    # 選擇病人
    if args.patient_id:
        if args.patient_id not in patient_ids:
            print(f"❌ Error: Patient {args.patient_id} not found")
            return
        target_patient = args.patient_id
    else:
        # 預設使用第一個病人
        target_patient = patient_ids[0]
    
    print(f"✓ Inference on patient: {target_patient}")
    
    # 建立 Dataset
    dataset = BraTSDataset(
        data_dir=config.DATA_DIR,
        patient_ids=[target_patient],
        image_size=config.IMAGE_SIZE,
        mode='val',
        use_smart_slice=True
    )
    
    # 載入模型
    model = AttentionUNet(
        n_channels=config.N_CHANNELS,
        n_classes=config.N_CLASSES,
        dropout_p=config.DROPOUT_P
    ).to(config.DEVICE)
    
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded (Dice: {checkpoint['dice']:.4f})")
    
    # 獲取影像
    image, mask = dataset[0]
    image = image.unsqueeze(0)  # Add batch dimension
    
    # MC Dropout 推論
    print(f"✓ Running MC Dropout inference ({config.MC_ITERATIONS} iterations)...")
    prediction, uncertainty = mc_dropout_inference(
        model=model,
        image_tensor=image,
        n_iterations=config.MC_ITERATIONS,
        device=config.DEVICE
    )
    
    # 視覺化
    output_dir = os.path.join(config.OUTPUT_DIR, 'inference')
    os.makedirs(output_dir, exist_ok=True)
    
    seg_path = os.path.join(output_dir, f'{target_patient}_segmentation.png')
    plot_results_with_uncertainty(
        image=image.squeeze(0).cpu().numpy(),
        mask=mask.squeeze(0).cpu().numpy(),
        prediction=prediction.squeeze(0),
        uncertainty=uncertainty.squeeze(0),
        save_path=seg_path,
        title=f"Segmentation Result - {target_patient}"
    )
    
    print(f"\n✅ Inference completed!")
    print(f"   Segmentation: {seg_path}")


def demo_command(args):
    """
    Demo 命令：使用少量資料跑 1 epoch 測試流程
    """
    print("\n" + "="*60)
    print("🎯 Brain MRI Segmentation - Demo Mode")
    print("="*60 + "\n")
    
    # 設定隨機種子
    config.set_seed()
    
    # 載入資料
    patient_ids = get_patient_ids(config.DATA_DIR)
    if not patient_ids:
        return
    
    # 只使用前 4 個病人
    demo_ids = patient_ids[:min(4, len(patient_ids))]
    train_ids = demo_ids[:3] if len(demo_ids) >= 3 else demo_ids
    val_ids = demo_ids[-1:] if len(demo_ids) >= 2 else demo_ids
    
    print(f"✓ Demo with {len(train_ids)} train + {len(val_ids)} val patients")
    
    # 建立 Dataset
    train_dataset = BraTSDataset(
        data_dir=config.DATA_DIR,
        patient_ids=train_ids,
        image_size=config.IMAGE_SIZE,
        mode='train',
        use_smart_slice=True
    )
    
    val_dataset = BraTSDataset(
        data_dir=config.DATA_DIR,
        patient_ids=val_ids,
        image_size=config.IMAGE_SIZE,
        mode='val',
        use_smart_slice=True
    )
    
    # 建立 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )
    
    # 建立模型
    model = AttentionUNet(
        n_channels=config.N_CHANNELS,
        n_classes=config.N_CLASSES,
        dropout_p=config.DROPOUT_P
    ).to(config.DEVICE)
    
    print(f"✓ Model initialized on {config.DEVICE}")
    
    # 建立訓練器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.DEVICE,
        use_amp=False  # Disable AMP for demo
    )
    
    # 訓練 1 epoch
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    trainer.train(epochs=1)
    
    print("\n✅ Demo completed!")


def main():
    """
    主函數
    """
    parser = argparse.ArgumentParser(
        description='Brain MRI Segmentation with Attention U-Net',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                    # Train the model
  python main.py infer                    # Infer on first patient
  python main.py infer --patient_id P001  # Infer on specific patient
  python main.py demo                     # Quick demo with 1 epoch
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    
    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--patient_id', type=str, help='Patient ID to infer')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run quick demo')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'infer':
        infer_command(args)
    elif args.command == 'demo':
        demo_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
