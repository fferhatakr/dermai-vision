import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sys
import os
import torch.nn.functional as F
import mlflow
import warnings

sys.path.append(os.getcwd())
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets
import hydra
from omegaconf import DictConfig

from src.engine.trainer_core import DermatologLightning
from src.dataloader.image_dataset import AlbumentationsDataset, Derm12345Dataset, get_album_transform
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):

    train_album, val_album = get_album_transform(
        image_size=cfg.model.image_size,
        mean=cfg.model.mean,
        std=cfg.model.std
    )
    mlflow.set_experiment(f"DermaScan_{cfg.model.backbone}")

    mode = cfg.train.mode

    if mode == "isic":

        print("ISIC K-Fold Training Mode")
        csv_path = cfg.paths.csv_path
        data_path = cfg.paths.data_path
        batch_size = cfg.train.batch_size
        epochs = cfg.train.epochs
        k_folds = cfg.train.k_folds
        backbone = cfg.model.backbone
        lr = cfg.train.learning_rate
        random_seed = cfg.train.random_seed
        patience = cfg.train.early_stop_patience
        exp_name = f"{cfg.train.experiment_name}_{backbone}"

        print("Reading Metadata")
        df = pd.read_csv(csv_path)
        df['image'] = df['image'].str.replace('_downsampled', '')
        df['lesion_id'] = df['lesion_id'].fillna(df['image'])
        df.set_index('image', inplace=True)

        df_nv = df[df['targets'] == 1]
        df_others = df[df['targets'] != 1]

        if len(df_nv) > 3000:
            df_nv = df_nv.sample(n=3000, random_state=random_seed)

        df = pd.concat([df_nv, df_others])

        full_dataset = datasets.ImageFolder(data_path)

        clean_file_names = []
        for f in full_dataset.imgs:
            raw_name = os.path.splitext(os.path.basename(f[0]))[0]
            clean_name = raw_name.replace('_downsampled', '')
            clean_file_names.append(clean_name)

        valid_imgfolder_indices = []
        valid_targets = []
        valid_groups = []
        skipped = 0

        for img_idx, name in enumerate(clean_file_names):
            if name in df.index:
                valid_imgfolder_indices.append(img_idx)
                valid_targets.append(df.loc[name, 'targets'])
                valid_groups.append(df.loc[name, 'lesion_id'])
            else:
                skipped += 1

        if skipped > 0:
            print(f"Warning: {skipped} The file could not be found in the CSV and was skipped")

        valid_imgfolder_indices = np.array(valid_imgfolder_indices)
        valid_targets = np.array(valid_targets)
        valid_groups = np.array(valid_groups)

        print(f"Total matching images: {len(valid_targets)}")
        print(f"Class distribution: {dict(zip(*np.unique(valid_targets, return_counts=True)))}")

        sgkf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(np.zeros(len(valid_targets)), valid_targets, groups=valid_groups)):
            train_g = set(valid_groups[train_idx])
            val_g = set(valid_groups[val_idx])
            overlap = train_g.intersection(val_g)

            
            if len(overlap) == 0:
                print("LEAK TEST SUCCESSFUL")
            else:
                print(f"Please note: {len(overlap)} There's a leak")
            

            real_train_indices = valid_imgfolder_indices[train_idx]
            real_val_indices = valid_imgfolder_indices[val_idx]

            dataset_train = AlbumentationsDataset(data_path, album_transform=train_album)
            dataset_val = AlbumentationsDataset(data_path, album_transform=val_album)

            train_sub = Subset(dataset_train, real_train_indices.tolist())
            val_sub = Subset(dataset_val, real_val_indices.tolist())

            curr_train_targets = valid_targets[train_idx]
            curr_val_targets = valid_targets[val_idx]

            print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")
            print(f"Train distribution: {dict(zip(*np.unique(curr_train_targets, return_counts=True)))}")
            print(f"Val distribution:   {dict(zip(*np.unique(curr_val_targets, return_counts=True)))}")

            sample_checks = min(20, len(real_train_indices))
            mismatch_count = 0
            for i in range(sample_checks):
                imgfolder_label = dataset_train.targets[real_train_indices[i]]
                metadata_label = curr_train_targets[i]
                if imgfolder_label != metadata_label:
                    mismatch_count += 1
            if mismatch_count > 0:
                print(f"LABEL Mismatch: {mismatch_count}/{sample_checks}")
                print(f"ImageFolder: {dataset_train.class_to_idx}")
            else:
                print("Label verification OK")

            counts = np.bincount(curr_train_targets)
            total = counts.sum()
            n_classes = len(counts)
            class_weights = torch.tensor(
                total / (n_classes * counts), dtype=torch.float32
            )

        
            class_weights = torch.clamp(class_weights, max=4.0)
            multipliers = cfg.train.class_multipliers_isic
            for i, mult in enumerate(multipliers):
                if i < len(class_weights):
                    class_weights[i] *= mult

            print("Class weights:", class_weights)  
            ws = 1. / np.sqrt(counts) 

            sample_weights = ws[curr_train_targets]
            sampler = WeightedRandomSampler(weights=sample_weights, 
                                            num_samples=len(sample_weights), replacement=True)

            train_loader = DataLoader(train_sub, batch_size=batch_size, sampler=sampler, num_workers=cfg.train.num_workers, pin_memory=True)
            val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True)

            model = DermatologLightning(
                backbone=backbone,
                lr=lr,
                max_epochs=epochs,
                class_weights=class_weights
            )

            checkpoint_callback = ModelCheckpoint(
                dirpath=cfg.paths.kfold_models_dir,
                filename=f"{exp_name}_fold_{fold+1}",
                monitor="val_loss", mode="min", save_top_k=1, verbose=True
            )
            early_stop = EarlyStopping(monitor="val_loss", patience=7, mode="min")
            lr_monitor = LearningRateMonitor(logging_interval='epoch')
            with mlflow.start_run(run_name=f"fold_{fold+1}"):
                    mlflow.log_params({
                        "backbone": backbone,
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "lr": lr,
                        "mel_weight_multiplier": 4.0,
                        "experiment": exp_name,
                    })

                    trainer = pl.Trainer(
                        max_epochs=epochs,
                        accelerator="cuda",
                        devices=1,
                        callbacks=[checkpoint_callback, early_stop, lr_monitor],
                        precision="bf16-mixed",
                        accumulate_grad_batches=4,
                        log_every_n_steps=10,
                    )

                    print(f"FOLD {fold+1} Starting training with {backbone}")
                    trainer.fit(model, train_loader, val_loader)
                    print(f"FOLD {fold+1} Completed!")

    elif mode == "finetune":
        print("DERM12345 Fine-Tuning Mode")
        train_dir = cfg.train.derm12345_train
        ft_checkpoint = cfg.train.fine_tune_checkpoint
        ft_lr = cfg.train.finetune_lr
        ft_batch = cfg.train.finetune_batch
        ft_accum = cfg.train.fine_tune_accum
        ft_epochs = cfg.train.finetune_epochs
        random_seed = cfg.train.random_seed
        patience = cfg.train.early_stop_patience

        full_derm = Derm12345Dataset(train_dir, album_transform=None)

        sss = StratifiedShuffleSplit(n_splits=1 , test_size=0.2, random_state=random_seed)
        train_idx, val_idx = next(sss.split(np.zeros(len(full_derm)), full_derm.targets))

        dataset_train = Derm12345Dataset(train_dir, album_transform=train_album, random_seed=cfg.train.random_seed)
        dataset_val = Derm12345Dataset(train_dir, album_transform=val_album,random_seed=cfg.train.random_seed)

        train_sub = Subset(dataset_train, train_idx.tolist())
        val_sub = Subset(dataset_val, val_idx.tolist())

        curr_train_targets = full_derm.targets[train_idx]
        curr_val_targets = full_derm.targets[val_idx]
        print(f"Fine-tune Train: {len(train_idx)} | Val: {len(val_idx)}")
        print(f"Train: {dict(zip(*np.unique(curr_train_targets, return_counts=True)))}")
        print(f"Val:   {dict(zip(*np.unique(curr_val_targets,   return_counts=True)))}")

        # The same logic as in the current ISIC code ÔÇö copied for consistency.
        # Inverse frequency: classes with fewer samples are given a higher weight.

        counts = np.bincount(curr_train_targets)
        total  = counts.sum()
        n_cls  = len(counts)
        class_weights = torch.tensor(total / (n_cls * counts), dtype=torch.float32)
        class_weights = torch.clamp(class_weights, max=4.0)

        multipliers = cfg.train.class_multipliers_finetune
        for i, mult in enumerate(multipliers):
            if i < len(class_weights):
                class_weights[i] *= mult 

        print("Fine-tune class weights:", class_weights)

        ws = 1. / np.sqrt(counts)
        sample_weights = ws[curr_train_targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_sub, batch_size=ft_batch,
            sampler=sampler, num_workers=cfg.train.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_sub, batch_size=ft_batch,
            shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True
        )

        model = DermatologLightning.load_from_checkpoint(
            ft_checkpoint,
            lr=ft_lr,
            max_epochs = ft_epochs,
            class_weights = class_weights,
        )
        print (f"Loaded Checkpoint:  {ft_checkpoint}")
        

        # Callbacks and Trainer
        ft_checkpoint = ModelCheckpoint(
            dirpath= "models/vision/",
            filename = f"DermScan_finetune_derm12345",
            monitor="val_loss", 
            mode="min", 
            save_top_k=1, 
            verbose=True
        )
        early_stop = EarlyStopping(monitor="val_loss", patience=7, mode="min")
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        with mlflow.start_run(run_name="finetune_derm12345"):
            mlflow.log_params({
                "mode": "finetune",
                "backbone": cfg.model.backbone,
                "finetune_lr": ft_lr,
                "finetune_epochs": ft_epochs,
                "batch_size": ft_batch,
                "checkpoint": ft_checkpoint,
            })

            trainer = pl.Trainer(
                max_epochs=ft_epochs,
                accelerator="cuda",
                devices=1,
                callbacks=[ft_checkpoint, early_stop, lr_monitor],
                precision="bf16-mixed",
                accumulate_grad_batches=ft_accum,
                log_every_n_steps=10,
            )
            print(f"Fine-tuning starting - {ft_epochs} epoch, LR = {ft_lr}")
            trainer.fit(model, train_loader, val_loader)
            print("Fine-tuning completed")
            print(f"Checkpoint saved: models/vision/DermScan_finetune_derm12345")



if __name__ == "__main__":
    main()


