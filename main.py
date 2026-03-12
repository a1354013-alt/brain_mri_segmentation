"""
Main CLI for Brain MRI Segmentation Project (v2.4 Final)
"""
import argparse
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from pathlib import Path

import config
from models import AttentionUNet
from utils import BraTSDataset, mc_dropout_inference, plot_results_with_uncertainty
from train import Trainer


def worker_init_fn(worker_id):
    """
    修正多 worker RNG 問題 (v2.4)：包含 torch seed
    """
    seed = config.RANDOM_SEED + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def get_patient_ids(data_dir: Path) -> list:
    if not data_dir.exists():
        print(f"❌ Error: Data directory not found at {data_dir}")
        return []
    return sorted([d.name for d in data_dir.iterdir() if d.is_dir()])


def train_command(args):
    print("\n🚀 Training Mode")
    config.set_seed()
    
    patient_ids = get_patient_ids(config.DATA_DIR)
    if len(patient_ids) == 0:
        print("❌ No data found in DATA_DIR. Please run download script first.")
        return

    rng = np.random.default_rng(config.RANDOM_SEED)
    rng.shuffle(patient_ids)
    
    split_idx = int(len(patient_ids) * config.TRAIN_VAL_SPLIT)
    train_ids = patient_ids[:split_idx]
    val_ids = patient_ids[split_idx:]
    
    train_dataset = BraTSDataset(config.DATA_DIR, train_ids, config.IMAGE_SIZE, mode='train')
    val_dataset = BraTSDataset(config.DATA_DIR, val_ids, config.IMAGE_SIZE, mode='val')
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=config.NUM_WORKERS, worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
        num_workers=config.NUM_WORKERS, worker_init_fn=worker_init_fn
    )
    
    model = AttentionUNet(config.N_CHANNELS, config.N_CLASSES, config.DROPOUT_P).to(config.DEVICE)
    
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader, device=config.DEVICE,
        output_dir=config.OUTPUT_DIR, checkpoint_path=config.CHECKPOINT_PATH, 
        model_state_path=config.MODEL_STATE_PATH, log_file=config.LOG_FILE,
        tensorboard_dir=config.TENSORBOARD_DIR
    )
    trainer.train(epochs=config.EPOCHS)


def infer_command(args):
    print("\n🔍 Inference Mode")
    
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
    if len(patient_ids) == 0:
        print("❌ No data found in DATA_DIR.")
        return
        
    # v2.4 體驗升級：自動挑選有效病人
    target_patient = args.patient_id
    dataset = None
    
    if target_patient:
        dataset = BraTSDataset(config.DATA_DIR, [target_patient], config.IMAGE_SIZE, mode='val')
        if len(dataset) == 0:
            print(f"⚠️  Patient {target_patient} is invalid. Searching for the first valid patient...")
            dataset = None
            
    if dataset is None:
        for pid in patient_ids:
            dataset = BraTSDataset(config.DATA_DIR, [pid], config.IMAGE_SIZE, mode='val')
            if len(dataset) > 0:
                target_patient = pid
                print(f"💡 Automatically selected valid patient: {target_patient}")
                break
    
    if dataset is None or len(dataset) == 0:
        print("❌ Error: No valid patients found in DATA_DIR.")
        return
        
    image, mask = dataset[0]
    
    prediction, uncertainty = mc_dropout_inference(
        model, image.unsqueeze(0), n_iterations=config.MC_ITERATIONS, method=args.uncertainty
    )
    
    save_path = config.OUTPUT_DIR / "inference" / f"{target_patient}_seg.png"
    plot_results_with_uncertainty(image.numpy(), mask.numpy(), prediction[0], uncertainty[0], save_path=save_path)
    print(f"✅ Saved to {save_path}")


def demo_command(args):
    print("\n🎯 Demo Mode")
    config.set_seed()
    
    patient_ids = get_patient_ids(config.DATA_DIR)
    if len(patient_ids) == 0:
        print("❌ No data found in DATA_DIR.")
        return
        
    # v2.4 Demo 模式也加入有效病人挑選
    demo_ids = []
    for pid in patient_ids:
        temp_ds = BraTSDataset(config.DATA_DIR, [pid], config.IMAGE_SIZE, mode='val')
        if len(temp_ds) > 0:
            demo_ids.append(pid)
        if len(demo_ids) >= 2:
            break
            
    if not demo_ids:
        print("❌ No valid patients found for Demo.")
        return
    
    train_dataset = BraTSDataset(config.DATA_DIR, demo_ids, config.IMAGE_SIZE, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=1, worker_init_fn=worker_init_fn)
    
    model = AttentionUNet(config.N_CHANNELS, config.N_CLASSES, config.DROPOUT_P).to(config.DEVICE)
    
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=train_loader, device=config.DEVICE,
        output_dir=config.DEMO_OUTPUT_DIR, checkpoint_path=config.DEMO_CHECKPOINT_PATH, 
        model_state_path=config.DEMO_MODEL_STATE_PATH, log_file=config.DEMO_LOG_FILE,
        tensorboard_dir=config.DEMO_TENSORBOARD_DIR, use_amp=False
    )
    trainer.train(epochs=1)
    
    # Demo 推論
    image, mask = train_dataset[0]
    prediction, uncertainty = mc_dropout_inference(model, image.unsqueeze(0), n_iterations=config.DEMO_MC_ITERATIONS)
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
