import os
import random
import logging
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import DiceLoss_Mask
from val import test_medical_image
from active_learning_strategies import ActiveLearningStrategyFactory
from datasets_skin import SemiSupervisedSkinDataset, ValSkinDataset, SkinTrainGenerator, SkinValGenerator


# Active Learning
class ActiveLearningManager:
    def __init__(self, pool_size=100, batch_size=6):
        self.pool_size = pool_size
        self.batch_size = batch_size
        self.sample_count = batch_size
        self.sampling_pool = set()
        self.labeled_indices = set()
        self.unlabeled_indices = set()
        self.strategy = ActiveLearningStrategyFactory.create_strategy('mask_difference')

    def initialize_pool(self, dataset_size):
        all_indices = list(range(dataset_size))
        self.unlabeled_indices = set(all_indices)
        self.labeled_indices = set()
        self.sampling_pool = set()

    def reset_sampling_pool(self):
        if len(self.unlabeled_indices) >= self.pool_size:
            self.sampling_pool = set(random.sample(list(self.unlabeled_indices), self.pool_size))
        else:
            self.sampling_pool = self.unlabeled_indices.copy()

    def select_samples(self, model, dataset, device, num_samples=None):
        self.reset_sampling_pool()
        available_indices = list(self.sampling_pool)
        samples_to_select = num_samples or self.sample_count
        samples_to_select = min(samples_to_select, len(available_indices))
        return self.strategy.select_samples(model, dataset, available_indices, samples_to_select, device)

    def update_labeled_status(self, selected_indices):
        for idx in selected_indices:
            if idx in self.unlabeled_indices:
                self.unlabeled_indices.remove(idx)
                self.labeled_indices.add(idx)
            if idx in self.sampling_pool:
                self.sampling_pool.remove(idx)


# Loss
def calc_loss_labeled(logits, labels, ce_loss, dice_loss, dice_weight=0.8):
    labels_processed = (labels > 0).long()
    loss_ce = ce_loss(logits, labels_processed)
    loss_dice = dice_loss(logits, labels_processed, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice

# Training
def trainer_al(args, model, snapshot_path, multimask_output):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device_ids = [int(i) for i in args.gpu.split(',')]
    main_device = f'cuda:{device_ids[0]}'
    model = model.to(main_device)

    ce_loss = CrossEntropyLoss(ignore_index=255)
    dice_loss = DiceLoss_Mask(args.num_classes + 1)

    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=0.01)
    writer = SummaryWriter(snapshot_path + '/log')

    # Dataset paths
    skin_data_root = args.data_root
    train_image_dir = os.path.join(skin_data_root, "Training_Data")
    val_image_dir = os.path.join(skin_data_root, "Validation_Data")

    train_image_list = [f for f in os.listdir(train_image_dir) if f.endswith(('.jpg', '.jpeg', '.png')) and '_superpixels' not in f]
    val_image_list = [f for f in os.listdir(val_image_dir) if f.endswith(('.jpg', '.jpeg', '.png')) and '_superpixels' not in f]

    # Active learning
    al_manager = ActiveLearningManager(pool_size=args.pool_size, batch_size=args.batch_size)
    db_train_full = SemiSupervisedSkinDataset(
        image_list=train_image_list,
        image_dir=train_image_dir,
        labeled_images={},
        mask_dir=os.path.join(skin_data_root, "Training_GroundTruth"),
        transform=transforms.Compose([SkinTrainGenerator(output_size=[args.img_size, args.img_size])])
    )
    al_manager.initialize_pool(len(db_train_full))

    db_val = ValSkinDataset(
        image_dir=val_image_dir,
        mask_dir=os.path.join(skin_data_root, "Validation_GroundTruth"),
        image_list=val_image_list[:20],
        transform=transforms.Compose([SkinValGenerator(output_size=[args.img_size, args.img_size])])
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False)

    # Training loop: each epoch selects via AL then trains; loss = avg(decoder1, decoder2)
    model.train()
    best_val_dice = -1.0
    best_ckpt_path = os.path.join(snapshot_path, 'best_model.pth')
    for epoch in tqdm(range(args.max_epochs), ncols=70):
        # AL: select top-k (batch_size) samples from pool
        selected_indices = al_manager.select_samples(model, db_train_full, main_device, args.batch_size)
        if not selected_indices:
            logging.info("No samples selected; stopping early.")
            break
        al_manager.update_labeled_status(selected_indices)
        print(f"[AL] epoch {epoch} selected {len(selected_indices)} samples: {selected_indices}")

        # Build one-off DataLoader for selected labeled samples
        selected_image_names = [db_train_full.image_list[i] for i in selected_indices]
        selected_labeled_images = {name: name for name in selected_image_names}
        db_train_selected = SemiSupervisedSkinDataset(
            image_list=db_train_full.image_list,
            image_dir=train_image_dir,
            labeled_images=selected_labeled_images,
            mask_dir=os.path.join(skin_data_root, "Training_GroundTruth"),
            transform=transforms.Compose([SkinTrainGenerator(output_size=[args.img_size, args.img_size])])
        )
        trainloader_iter = DataLoader(Subset(db_train_selected, selected_indices), batch_size=args.batch_size, shuffle=False)
        # Ensure training mode (strategy may set eval)
        model.train()

        # Supervised train (average loss of two decoders)
        for sampled_batch in trainloader_iter:
            image_batch = sampled_batch['image'].to(main_device)
            label_batch = sampled_batch['label'].to(main_device)

            outputs = model(image_batch, multimask_output, args.img_size, -1, args.promptmode)
            logits1 = outputs['low_res_logits1']
            logits2 = outputs['low_res_logits2']

            loss1, loss1_ce, loss1_dice = calc_loss_labeled(logits1, label_batch, ce_loss, dice_loss)
            loss2, loss2_ce, loss2_dice = calc_loss_labeled(logits2, label_batch, ce_loss, dice_loss)
            loss = 0.5 * (loss1 + loss2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            writer.add_scalar('train/loss_avg', loss.item(), epoch)
            writer.add_scalar('train/loss_dec1', loss1.item(), epoch)
            writer.add_scalar('train/loss_dec2', loss2.item(), epoch)
            writer.add_scalar('train/loss_ce_dec1', loss1_ce.item(), epoch)
            writer.add_scalar('train/loss_dice_dec1', loss1_dice.item(), epoch)
            writer.add_scalar('train/loss_ce_dec2', loss2_ce.item(), epoch)
            writer.add_scalar('train/loss_dice_dec2', loss2_dice.item(), epoch)
            logging.info(f"[TRAIN] epoch {epoch} - loss_dec1: {loss1.item():.4f}, loss_dec2: {loss2.item():.4f}, loss_avg: {loss.item():.4f}")
            print(f"[TRAIN] epoch {epoch} - loss_dec1: {loss1.item():.4f}, loss_dec2: {loss2.item():.4f}, loss_avg: {loss.item():.4f}")

        # Validation
        model.eval()
        metrics = []
        with torch.no_grad():
            for sampled_batch in valloader:
                img = sampled_batch['image'].to(main_device)
                lab = sampled_batch['label'].to(main_device)
                m = test_medical_image(img, lab, model, classes=args.num_classes + 1)
                metrics.append(m)
        # Compute mean Dice over foreground classes across validation set
        dice_scores = []
        for sample_metrics in metrics:
            # sample_metrics: list of (dice, hd95) per class, including background at index 0
            fg_dices = [cls_metrics[0] for cls_idx, cls_metrics in enumerate(sample_metrics) if cls_idx > 0]
            if len(fg_dices) > 0:
                dice_scores.append(float(np.mean(fg_dices)))
            else:
                dice_scores.append(0.0)
        epoch_dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
        logging.info(f"Epoch {epoch} validation Dice (fg mean): {epoch_dice:.6f}")
        # Save best checkpoint starting from epoch 10 (1-based)
        if (epoch + 1) >= 10 and epoch_dice > best_val_dice:
            best_val_dice = epoch_dice
            best_ckpt_path_dice = os.path.join(snapshot_path, f"best_model_dice{best_val_dice:.4f}.pth")
            torch.save(model.state_dict(), best_ckpt_path_dice)
            logging.info(f"Saved best model to {best_ckpt_path_dice} (Dice={best_val_dice:.6f}, epoch={epoch})")
        model.train()


# ================= Main =================
if __name__ == "__main__":
    import argparse, os, random, numpy as np, torch
    import torch.backends.cudnn as cudnn
    from importlib import import_module
    from hydra.core.global_hydra import GlobalHydra
    from sam2.build_sam import build_sam2  

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../data/skin_lesion_dataset", help="Root path of dataset")
    parser.add_argument("--output", type=str, default="./output", help="Output path")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--deterministic", action="store_true", help="Deterministic training")
    parser.add_argument("--base_lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pool_size", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--promptmode", type=int, default=0)
    parser.add_argument("--module", type=str, default="sam_lora_image_encoder_prompt", help="Python module containing LoRA_Sam")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lora_ckpt", type=str, default=None)
    args = parser.parse_args()

    # set random seeds
    if args.deterministic:
        cudnn.deterministic, cudnn.benchmark = True, False
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # output folder
    snapshot_path = os.path.join(args.output, f"Skin_{args.img_size}")
    os.makedirs(snapshot_path, exist_ok=True)

    # load SAM2
    model_cfg = "./sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
    assert os.path.exists(model_cfg) and os.path.exists(checkpoint)

    GlobalHydra.instance().clear()
    sam = build_sam2(model_cfg, checkpoint)

    # wrap with LoRA
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).to(f"cuda:{args.gpu}")

    if args.lora_ckpt:
        net.load_lora_parameters(args.lora_ckpt)

    # save config
    with open(os.path.join(snapshot_path, 'config.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # run warmup training
    trainer_al(args, net, snapshot_path, multimask_output=True)

