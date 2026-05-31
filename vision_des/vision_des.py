"""
Vision-based Dynamic Ensemble Selection (VisionDES).

This module implements a dynamic ensemble selection framework
for image classification using DINO embeddings and FAISS-based
region-of-competence retrieval.
"""

import faiss
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn

from torch.nn.functional import cosine_similarity, softmax
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm


EPSILON = 1e-6
WEIGHT_EPSILON = 1e-8

MIN_ALPHA = 0.2
MAX_ALPHA = 0.8

DEFAULT_BATCH_SIZE = 32


# Visualization utilities 
def visualize_test_and_roc(
    test_img,
    roc_imgs,
    local_labels,
    distances=None,
):
    """
    Visualize a test image and its region of competence.

    Parameters
    ----------
    test_img : torch.Tensor
        Query image.
    roc_imgs : torch.Tensor
        Retrieved region-of-competence images.
    local_labels : np.ndarray
        Labels of RoC samples.
    """

    def denormalize(img, mean, std):
        mean = torch.tensor(mean, device=img.device).view(-1, 1, 1)
        std = torch.tensor(std, device=img.device).view(-1, 1, 1)
        return img * std + mean

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    k = roc_imgs.size(0)
    ncols = min(k, 5)
    nrows = 1 + (k + ncols - 1) // ncols

    plt.figure(figsize=(3 * ncols, 3 * nrows))

    plt.subplot(nrows, ncols, 1)
    img = denormalize(test_img, mean, std).clamp(0, 1)
    plt.imshow(TF.to_pil_image(img.cpu()))
    plt.title("Test Image")
    plt.axis("off")

    for i in range(k):
        plt.subplot(nrows, ncols, i + 2)
        img = denormalize(roc_imgs[i], mean, std).clamp(0, 1)
        plt.imshow(TF.to_pil_image(img.cpu()))
        title = f"RoC #{i + 1} | Label {local_labels[i]}"
        if distances is not None:
            title += f"\nDist={distances[i]:.3f}"
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# Model introspection utilities 
def get_last_linear_layer(model: nn.Module) -> nn.Linear:
    """
    Return the last linear layer of a model.
    """
    
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):
            return module
    raise RuntimeError("No Linear layer found.")


def get_features_before_last_linear(model, x):
    """
    Extract features immediately before the final linear layer.
    """
    
    features = {}
    last_linear = get_last_linear_layer(model)

    def hook(_, inp, __):
        features["feat"] = inp[0].detach()

    handle = last_linear.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        _ = model(x)
    handle.remove()

    return features["feat"]


# FIRE criterion
def fire_check(
    local_labels: np.ndarray,
    preds: np.ndarray,
    per_class_min: int = 1,
) -> bool:
    """
    Check whether each class in the local region
    has at least the required number of correct predictions.
    """
    
    for c in np.unique(local_labels):
        mask = local_labels == c
        if np.sum(preds[mask] == c) < per_class_min:
            return False
    return True


# VisionDES
class VisionDES:
    """
    Vision-based Dynamic Ensemble Selection.

    Uses DINO embeddings and FAISS nearest-neighbor search
    to define a region of competence and dynamically
    weight ensemble members.
    """
    
    def __init__(self, dsel_dataset, pool, device):
        self.device = device
        self.pool = pool
        self.dsel_dataset = dsel_dataset
        self.suspected_model_votes = []

        self.dsel_loader = DataLoader(
            dsel_dataset,
            batch_size=DEFAULT_BATCH_SIZE,
            shuffle=False,
        )

        self.dino_model = (
            timm.create_model("vit_base_patch16_224.dino", pretrained=True)
            .to(self.device)
            .eval()
        )


    def fit(self) -> None:
        """
        Extract DINO embeddings from the DSEL set and build
        a FAISS index for region-of-competence retrieval.
        """
        
        embs, labels = [], []

        with torch.no_grad():
            for x, y in tqdm(self.dsel_loader, desc="Extracting embeddings"):
                x = x.to(self.device)
                features = self.dino_model.forward_features(x)[:, 0, :].cpu()
                embs.append(features)
                labels.append(y)

        self.dsel_embeddings = torch.cat(embs).numpy().astype("float32")
        self.dsel_labels = torch.cat(labels).numpy()

        self.index = faiss.IndexFlatL2(self.dsel_embeddings.shape[1])
        self.index.add(self.dsel_embeddings)


    def predict(
        self,
        test_img: torch.Tensor,
        k: int = 7,
        explain: bool = False,
        knora_e: bool = False,
        top: bool = False,
        n: int = 3,
        use_fire: bool = False,
    ) -> int:
        """
        Predict the class label for a test image using
        dynamic ensemble selection.
        """

        model_records = []

        # ---------- Query ----------
        with torch.no_grad():
            emb = (
                self.dino_model.forward_features(test_img.unsqueeze(0).to(self.device))[
                    :, 0, :
                ]
                .cpu()
                .numpy()
                .astype("float32")
            )

        distances, neighbors = self.index.search(emb, k)
        idxs = neighbors[0]
        distances = distances[0]
        
        roc_imgs = torch.stack([self.dsel_dataset[i][0] for i in idxs]).to(self.device)
        local_labels = self.dsel_labels[idxs]

        competences, sims, probs = [], [], []

        for clf in self.pool:
            clf.eval()
            with torch.no_grad():
                out = clf(roc_imgs)
                preds = out.argmax(1).cpu().numpy()
                correct = int((preds == local_labels).sum())

                # ---------- KNORA-E ----------
                if knora_e and correct < k:
                    continue

                competence = correct / k
                fire_ok = fire_check(local_labels, preds) if use_fire else True

                logit = clf(test_img.unsqueeze(0).to(self.device)).squeeze(0)
                prob = softmax(logit, dim=0)

                test_features = get_features_before_last_linear(
                    clf,
                    test_img.unsqueeze(0).to(self.device),
                )
                
                roc_features = get_features_before_last_linear(
                    clf,
                    roc_imgs,
                ).mean(0, keepdim=True)
                
                sim = cosine_similarity(
                    test_features,
                    roc_features,
                ).item()

                competences.append(competence)
                sims.append(sim)
                probs.append(prob)

                model_records.append(
                    {
                        "model": clf,
                        "competence": competence,
                        "correct": correct,
                        "fire": fire_ok,
                        "sim": sim,
                        "prob": prob,
                    }
                )

        if not probs:
            logits = []
            for clf in self.pool:
                with torch.no_grad():
                    logits.append(clf(test_img.unsqueeze(0).to(self.device)).squeeze(0))
            return torch.stack(logits).mean(0).argmax().item()

        competences = np.array(competences)
        sims = np.array(sims)

        if top:
            ranking = np.argsort(-competences)[:n]
            competences = competences[ranking]
            sims = sims[ranking]
            probs = [probs[i] for i in ranking]
            model_records = [model_records[i] for i in ranking]

        var_c = np.var(competences)
        var_s = np.var(sims)

        mean_c = np.mean(competences)
        mean_s = np.mean(sims)

        rel_var_c = var_c / (mean_c + EPSILON)
        rel_var_s = var_s / (mean_s + EPSILON)

        alpha = float(
            np.clip(
                rel_var_c / (rel_var_c + rel_var_s + EPSILON),
                MIN_ALPHA,
                MAX_ALPHA,
            )
        )

        scores = alpha * competences + (1 - alpha) * sims
        weights = scores / (scores.sum() + WEIGHT_EPSILON)

        weighted_logits = torch.zeros_like(probs[0])
        for w, p in zip(weights, probs):
            weighted_logits += w * p

        self.suspected_model_votes.append(int(np.argmin(sims)))

        if explain:
            print("\n========== EXPLAINABILITY ==========")
            print(f"KNORA-E enabled: {knora_e}")
            print(f"alpha: {alpha:.3f}")
            print("-" * 50)

            for i, r in enumerate(model_records):
                pred = r["prob"].argmax().item()
                conf = r["prob"][pred].item()
                
                top_probs, top_classes = torch.topk(
                    r["prob"],
                    k=min(5, len(r["prob"]))
                )
                
                print(f"Model {i}: {r['model'].__class__.__name__}")
                print(f"  Competence : {r['competence']:.3f}")
                print(f"  Correct    : {r['correct']}/{k}")
                print(f"  Similarity : {r['sim']:.3f}")
                print(f"  FIRE pass  : {r['fire']}")
                print(f"  Weight     : {weights[i]:.3f}")
                print(f"  Prediction : {pred} (conf {conf:.3f})")
                
                print("  Top-5 Predictions:")
                for cls, prob in zip(top_classes.tolist(), top_probs.tolist()):
                    print(f"    Class {cls:<3} -> {prob:.4f}")
                
                print("-" * 50)

            final_probs, final_classes = torch.topk(
                weighted_logits,
                k=min(5, len(weighted_logits))
            )
            
            print("\nFinal Ensemble Top-5:")
            for cls, prob in zip(
                final_classes.tolist(),
                final_probs.tolist(),
            ):
                print(f"  Class {cls:<3} -> {prob:.4f}")

            visualize_test_and_roc(test_img, roc_imgs, local_labels, distances)

        return weighted_logits.argmax().item()
