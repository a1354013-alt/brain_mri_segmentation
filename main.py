"""
Main CLI for Brain MRI Segmentation Project
"""
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

import config
from models import AttentionUNet
from utils import BraTSDataset, mc_dropout_inference, plot_results_with_uncertainty
from train import Trainer


def get_patient_ids(data_dir: Path) -> list:
    if not data_dir.exists():
        print(f"❌ Error: Data directory not found at {data_dir}")
        return []
    return sorted([d.name for d in data_dir.iterdir() if d.is_dir()])


def train_command(args):
    print("\n🚀 Training Mode")
    config.set_seed()
    
    patient_ids = get_patient_ids(config.DATA_DIR)
    if not patient_ids: return

    # 2. Shuffle 且可重現
    rng = np.random.default_rng(config.RANDOM_SEED)
    rng.shuffle(patient_ids)
    
    split_idx = int(len(patient_ids) * config.TRAIN_VAL_SPLIT)
    train_ids = patient_ids[:split_idx]
    val_ids = patient_ids[split_idx:]
    
    train_dataset = BraTSDataset(config.DATA_DIR, train_ids, config.IMAGE_SIZE, mode='train')
    val_dataset = BraTSDataset(config.DATA_DIR, val_ids, config.IMAGE_SIZE, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    model = AttentionUNet(config.N_CHANNELS, config.N_CLASSES, config.DROPOUT_P).to(config.DEVICE)
    
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader, device=config.DEVICE,
        output_dir=config.OUTPUT_DIR, checkpoint_path=config.CHECKPOINT_PATH, model_state_path=config.MODEL_STATE_PATH
    )
    trainer.train(epochs=config.EPOCHS)


def infer_command(args):
    print("\n🔍 Inference Mode")
    
    # 4. Checkpoint 讀取邏輯
    model = AttentionUNet(config.N_CHANNELS, config.N_CLASSES, config.DROPOUT_P).to(config.DEVICE)
    
    if config.MODEL_STATE_PATH.exists():
        model.load_state_dict(torch.load(config.MODEL_STATE_PATH, map_location=config.DEVICE))
        print(f"✓ Loaded model state from {config.MODEL_STATE_PATH}")
    elif config.CHECKPOINT_PATH.exists():
        checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded model state from checkpoint {config.CHECKPOINT_PATH}")
    else:
        print("❌ Error: No model found. Please train first.")
        return

    patient_ids = get_patient_ids(config.DATA_DIR)
    target_patient = args.patient_id if args.patient_id else patient_ids[0]
    
    dataset = BraTSDataset(config.DATA_DIR, [target_patient], config.IMAGE_SIZE, mode='val')
    image, mask = dataset[0]
    
    prediction, uncertainty = mc_dropout_inference(model, image.unsqueeze(0), method=args.uncertainty)
    
    save_path = config.OUTPUT_DIR / "inference" / f"{target_patient}_seg.png"
    plot_results_with_uncertainty(image.numpy(), mask.numpy(), prediction[0], uncertainty[0], save_path=save_path)
    print(f"✅ Saved to {save_path}")


def demo_command(args):
    print("\n🎯 Demo Mode")
    config.set_seed()
    
    patient_ids = get_patient_ids(config.DATA_DIR)
    demo_ids = patient_ids[:min(2, len(patient_ids))]
    
    # 3. Demo 模式隔離
    train_dataset = BraTSDataset(config.DATA_DIR, demo_ids, config.IMAGE_SIZE, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=1)
    
    model = AttentionUNet(config.N_CHANNELS, config.N_CLASSES, config.DROPOUT_P).to(config.DEVICE)
    
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=train_loader, device=config.DEVICE,
        output_dir=config.DEMO_OUTPUT_DIR, checkpoint_path=config.DEMO_CHECKPOINT_PATH, model_state_path=config.DEMO_MODEL_STATE_PATH,
        use_amp=False
    )
    trainer.train(epochs=1)
    
    # Demo 推論
    image, mask = train_dataset[0]
    prediction, uncertainty = mc_dropout_inference(model, image.unsqueeze(0))
    save_path = config.DEMO_OUTPUT_DIR / "demo_inference.png"
    plot_results_with_uncertainty(image.numpy(), mask.numpy(), prediction[0], uncertainty[0], save_path=save_path)
    print(f"✅ Demo completed. Results in {config.DEMO_OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description='Brain MRI Segmentation')
    subparsers = parser.add_subparsers(dest='command')
    
    subparsers.add_parser('train')
    
    infer_p = subparsers.add_parser('infer')
    infer_p.add_argument('--patient_id', type=str)
    infer_p.add_argument('--uncertainty', choices=['var', 'entropy'], default='var')
    
    subparsers.add_parser('demo')
    
    args = parser.parse_args()
    if args.command == 'train': train_command(args)
    elif args.command == 'infer': infer_command(args)
    elif args.command == 'demo': demo_command(args)
    else: parser.print_help()


if __name__ == "__main__":
    main()
