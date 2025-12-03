from nms import non_max_suppression
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from process_crops import cropDataLoader
from model import get_vgg16_model
from IoU import intersection_over_union

from copy import deepcopy

# We define the training as a function so we can easily re-use it.
def train(model, optimizer, trainset, train_loader, valset, val_loader, device: torch.device, num_epochs=10):

    out_dict = {
        "train_loss": {"class": [], "bbox": [], "total": []},
        "val_loss": {"class": [], "bbox": [], "total": []},
        "train_acc": {"class": [], "bbox": []},   # bbox = mean IoU
        "val_acc": {"class": [], "bbox": []}
    }

    best_acc = 0
    best_model = deepcopy(model)

    for epoch in tqdm(range(num_epochs), unit="epoch"):
        model.train()

        # Running metrics
        running_cls_loss = 0.0
        running_bbox_loss = 0.0
        running_total_loss = 0.0

        running_correct = 0
        running_samples = 0

        running_iou_sum = 0.0
        running_iou_count = 0

        # ─────────────────────────────────────────────────────────
        # TRAIN LOOP
        # ─────────────────────────────────────────────────────────
        for images, class_targets, bbox_targets, bbox_proposed in tqdm(
            train_loader, leave=False, total=len(train_loader), desc="Training"
        ):
            images = images.to(device)
            class_targets = class_targets.to(device)
            bbox_targets = bbox_targets.to(device)
            bbox_proposed = bbox_proposed.to(device)

            optimizer.zero_grad()

            class_logits, bbox_pred_offsets = model(images)

            # Classification loss
            cls_loss = F.cross_entropy(class_logits, class_targets)

            # Positive samples only (not background)
            pos_mask = (class_targets != 0)

            # Regression targets
            if pos_mask.any():
                g_cx, g_cy, g_w, g_h = bbox_targets[:, 0], bbox_targets[:, 1], bbox_targets[:, 2], bbox_targets[:, 3]
                p_cx, p_cy, p_w, p_h = bbox_proposed[:, 0], bbox_proposed[:, 1], bbox_proposed[:, 2], bbox_proposed[:, 3]

                # Compute target offsets
                t_x = (g_cx - p_cx) / p_w
                t_y = (g_cy - p_cy) / p_h
                t_w = torch.log(g_w / p_w)
                t_h = torch.log(g_h / p_h)

                target_offsets = torch.stack([t_x, t_y, t_w, t_h], dim=1)

                # Regression loss
                bbox_loss = F.smooth_l1_loss(
                    bbox_pred_offsets[pos_mask],
                    target_offsets[pos_mask]
                )

                # Compute IoU for accuracy
                tx_pred, ty_pred, tw_pred, th_pred = (
                    bbox_pred_offsets[:, 0],
                    bbox_pred_offsets[:, 1],
                    bbox_pred_offsets[:, 2],
                    bbox_pred_offsets[:, 3],
                )

                pred_cx = tx_pred * p_w + p_cx
                pred_cy = ty_pred * p_h + p_cy
                pred_w = torch.exp(tw_pred) * p_w
                pred_h = torch.exp(th_pred) * p_h

                pred_box = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=1)
                gt_box = torch.stack([g_cx, g_cy, g_w, g_h], dim=1)

                iou = intersection_over_union(pred_box[pos_mask], gt_box[pos_mask])
                running_iou_sum += iou.sum().item()
                running_iou_count += iou.numel()

            else:
                bbox_loss = torch.tensor(0.0, device=device)

            # Total loss
            loss = cls_loss + 0.5 * bbox_loss
            loss.backward()
            optimizer.step()

            # Update running losses
            running_cls_loss += cls_loss.item() * images.size(0)
            running_bbox_loss += bbox_loss.item() * images.size(0)
            running_total_loss += loss.item() * images.size(0)

            # Classification accuracy
            preds = class_logits.argmax(1)
            running_correct += (preds == class_targets).sum().item()
            running_samples += class_targets.size(0)

        # Epoch metrics
        epoch_cls_loss = running_cls_loss / running_samples
        epoch_bbox_loss = running_bbox_loss / running_samples
        epoch_total_loss = running_total_loss / running_samples

        epoch_cls_acc = running_correct / running_samples
        epoch_bbox_acc = (running_iou_sum / running_iou_count) if running_iou_count > 0 else 0.0

        out_dict["train_loss"]["class"].append(epoch_cls_loss)
        out_dict["train_loss"]["bbox"].append(epoch_bbox_loss)
        out_dict["train_loss"]["total"].append(epoch_total_loss)

        out_dict["train_acc"]["class"].append(epoch_cls_acc)
        out_dict["train_acc"]["bbox"].append(epoch_bbox_acc)

        # ─────────────────────────────────────────────────────────
        # VALIDATION LOOP
        # ─────────────────────────────────────────────────────────
        model.eval()

        val_running_cls_loss = 0.0
        val_running_bbox_loss = 0.0
        val_running_total_loss = 0.0

        val_running_correct = 0
        val_running_samples = 0

        val_running_iou_sum = 0.0
        val_running_iou_count = 0

        with torch.no_grad():
            for images, class_targets, bbox_targets, bbox_proposed in tqdm(
                val_loader, leave=False, total=len(val_loader), desc="Validating"
            ):
                images = images.to(device)
                class_targets = class_targets.to(device)
                bbox_targets = bbox_targets.to(device)
                bbox_proposed = bbox_proposed.to(device)

                class_logits, bbox_pred_offsets = model(images)

                # classification
                cls_loss = F.cross_entropy(class_logits, class_targets)

                # bbox regression (same logic)
                pos_mask = (class_targets != 0)

                if pos_mask.any():
                    g_cx, g_cy, g_w, g_h = bbox_targets[:, 0], bbox_targets[:, 1], bbox_targets[:, 2], bbox_targets[:, 3]
                    p_cx, p_cy, p_w, p_h = bbox_proposed[:, 0], bbox_proposed[:, 1], bbox_proposed[:, 2], bbox_proposed[:, 3]

                    t_x = (g_cx - p_cx) / p_w
                    t_y = (g_cy - p_cy) / p_h
                    t_w = torch.log(g_w / p_w)
                    t_h = torch.log(g_h / p_h)

                    target_offsets = torch.stack([t_x, t_y, t_w, t_h], dim=1)

                    bbox_loss = F.smooth_l1_loss(
                        bbox_pred_offsets[pos_mask],
                        target_offsets[pos_mask]
                    )

                    # IoU
                    tx_pred, ty_pred, tw_pred, th_pred = (
                        bbox_pred_offsets[:, 0],
                        bbox_pred_offsets[:, 1],
                        bbox_pred_offsets[:, 2],
                        bbox_pred_offsets[:, 3],
                    )

                    pred_cx = tx_pred * p_w + p_cx
                    pred_cy = ty_pred * p_h + p_cy
                    pred_w = torch.exp(tw_pred) * p_w
                    pred_h = torch.exp(th_pred) * p_h

                    pred_box = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=1)
                    gt_box = torch.stack([g_cx, g_cy, g_w, g_h], dim=1)

                    iou = intersection_over_union(pred_box[pos_mask], gt_box[pos_mask])

                    val_running_iou_sum += iou.sum().item()
                    val_running_iou_count += iou.numel()
                else:
                    bbox_loss = torch.tensor(0.0, device=device)

                total_loss = cls_loss + 0.5 * bbox_loss

                # accumulate loss
                val_running_cls_loss += cls_loss.item() * images.size(0)
                val_running_bbox_loss += bbox_loss.item() * images.size(0)
                val_running_total_loss += total_loss.item() * images.size(0)

                # accuracy
                preds = class_logits.argmax(1)
                val_running_correct += (preds == class_targets).sum().item()
                val_running_samples += class_targets.size(0)

        # Epoch validation metrics
        val_cls_loss = val_running_cls_loss / val_running_samples
        val_bbox_loss = val_running_bbox_loss / val_running_samples
        val_total_loss = val_running_total_loss / val_running_samples

        val_cls_acc = val_running_correct / val_running_samples
        val_bbox_acc = (val_running_iou_sum / val_running_iou_count) if val_running_iou_count > 0 else 0.0

        out_dict["val_loss"]["class"].append(val_cls_loss)
        out_dict["val_loss"]["bbox"].append(val_bbox_loss)
        out_dict["val_loss"]["total"].append(val_total_loss)

        out_dict["val_acc"]["class"].append(val_cls_acc)
        out_dict["val_acc"]["bbox"].append(val_bbox_acc)

        # best model tracking
        if val_cls_acc > best_acc:
            best_acc = val_cls_acc
            best_model = deepcopy(model)

        tqdm.write(
            f"Epoch {epoch+1} | "
            f"Train cls_loss={epoch_cls_loss:.3f}, bbox_loss={epoch_bbox_loss:.3f}, IoU={epoch_bbox_acc:.3f}, Acc={epoch_cls_acc:.3f} | "
            f"Val cls_loss={val_cls_loss:.3f}, bbox_loss={val_bbox_loss:.3f}, IoU={val_bbox_acc:.3f}, Acc={val_cls_acc:.3f}"
        )

    return out_dict, best_model

def plot_training(training_dict: dict):
    def stylize(ax):
        ax.legend()
        ax.grid(linestyle="--")
        ax.spines[['right', 'top']].set_visible(False)
        ax.xaxis.set_major_locator(MultipleLocator(1))

    epochs = range(1, len(training_dict["train_loss"]["total"]) + 1)

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(16, 6))

    # ───────────────────────────────────────────────────────────
    # ACCURACY PLOTS (class accuracy + IoU accuracy)
    # ───────────────────────────────────────────────────────────
    ax_acc.set_title("Accuracy Metrics")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy / IoU")

    # Training metrics
    ax_acc.plot(
        epochs, training_dict["train_acc"]["class"],
        label="Train Class Acc"
    )
    ax_acc.plot(
        epochs, training_dict["train_acc"]["bbox"],
        label="Train IoU (BBox Acc)"
    )

    # Validation metrics
    ax_acc.plot(
        epochs, training_dict["val_acc"]["class"],
        label="Val Class Acc",
        linestyle="--"
    )
    ax_acc.plot(
        epochs, training_dict["val_acc"]["bbox"],
        label="Val IoU (BBox Acc)",
        linestyle="--"
    )

    stylize(ax_acc)

    # ───────────────────────────────────────────────────────────
    # LOSS PLOTS (class loss + bbox loss + total)
    # ───────────────────────────────────────────────────────────
    ax_loss.set_title("Loss Metrics")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")

    # Training losses
    ax_loss.plot(
        epochs, training_dict["train_loss"]["class"],
        label="Train Class Loss"
    )
    ax_loss.plot(
        epochs, training_dict["train_loss"]["bbox"],
        label="Train BBox Loss"
    )
    ax_loss.plot(
        epochs, training_dict["train_loss"]["total"],
        label="Train Total Loss"
    )

    # Validation losses
    ax_loss.plot(
        epochs, training_dict["val_loss"]["class"],
        label="Val Class Loss",
        linestyle="--"
    )
    ax_loss.plot(
        epochs, training_dict["val_loss"]["bbox"],
        label="Val BBox Loss",
        linestyle="--"
    )
    ax_loss.plot(
        epochs, training_dict["val_loss"]["total"],
        label="Val Total Loss",
        linestyle="--"
    )

    stylize(ax_loss)

    fig.suptitle("Training & Validation Metrics", fontsize=16)
    fig.tight_layout()

    plt.savefig("figures/all_metrics.png", dpi=200)
    plt.savefig("figures/all_metrics.pdf")
    plt.show()

def save_model(model):
    while True:
        save = input(f"Do you want to save the model? [y/n]\n")
        if save.lower() == 'y':
            break
        elif save.lower() == 'n':
            return
        else:
            print("Please enter y or n.")

    files = [file.split('\\')[-1] for file in glob.glob('models/*.pth')]
    print(f'Current model files:')
    for file in files:
        print(f'\t{file}')

    name = input('Enter file name: ')

    torch.save(model.state_dict(), f'models/{name}.pth')

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_vgg16_model(pretrained=True)
    model.to(device)

    transforms = [transforms.RandomRotation(30), transforms.RandomVerticalFlip(0.25), transforms.RandomHorizontalFlip(0.25)]
    (train_loader, val_loader, _), (trainset, valset, _) = cropDataLoader(batch_size=64, transform=transforms, train_ratio=0.9, val_ratio=0.1, test_ratio=0)

    optimizer = torch.optim.Adam(model.parameters())
    out_dict, best_model = train(model, optimizer, trainset, train_loader, valset, val_loader, device=device, num_epochs=10)

    plot_training(out_dict)

    save_model(best_model)