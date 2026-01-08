import os
import gc
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple, List
from DropletCorLab import save_pickle
from DropletCorLab.preprocessing.aligment import estimate_translation
from DropletCorLab.preprocessing import image_preprocessing
from DropletCorLab.features.statistical.sampling import build_cropped_interpolated_roi
from DropletCorLab.features.statistical.builders.frame_feature_export import filter_edge_one_side
from DropletCorLab.features.learned.training.train_conv_ae import train_ae_conv_fc
from DropletCorLab.features.learned.encoders.conv_ae import AEConvFC
from DropletCorLab.preprocessing import image_preprocessing


class AEDualROIFeatureBuilder:
    """
    Per-frame, per-droplet AE feature builder.

    Mirrors the statistical feature builder logic, but replaces
    color descriptors with AE latent representations.
    """

    def __init__(
        self,
        model_inner,
        model_outer,
        device: torch.device,
        target_size: Tuple[int, int] = (32, 32),
        batch_size: int = 32,
    ):
        self.model_inner = model_inner.to(device).eval()
        self.model_outer = model_outer.to(device).eval()
        self.device = device
        self.target_size = target_size
        self.batch_size = batch_size

    def _encode_batch(self, model, tensor: torch.Tensor) -> np.ndarray:
        latents = []
        with torch.no_grad():
            for i in range(0, tensor.size(0), self.batch_size):
                batch = tensor[i : i + self.batch_size].to(self.device)
                z = model.encode(batch)
                latents.append(z.cpu())
        return torch.cat(latents, dim=0).numpy()


    def build_per_frame_features(
        self,
        image_rgb_raw: np.ndarray,
        frame_ID: int,
        image_pos_ref: np.ndarray,
        image_bri_ref: np.ndarray,
        mask_coords_inner: Dict[str, list],
        mask_coords_outer: Dict[str, list],
        geos_all: Dict[str, Dict],
        edge_thresh: int = 25,
        save: bool = True,
        save_dir: str = "",
    ) -> Dict:
        """
        Build AE features for all droplets in ONE frame.
        """

        # --- alignment & brightness correction ---
        edge_x, edge_y = estimate_translation(image_pos_ref, image_rgb_raw)
        edge_x = min(edge_x, -edge_thresh) if edge_x < 0 else max(edge_x, edge_thresh)
        edge_y = min(edge_y, -edge_thresh) if edge_y < 0 else max(edge_y, edge_thresh)

        image_rgb = image_preprocessing(image_rgb_raw, image_pos_ref, image_bri_ref)
        image_shape = image_rgb.shape

        identifiers = {}
        inner_imgs, outer_imgs = [], []
        geo_mat = []

        valid_bboxes = []

        # --- per-droplet loop (mirrors statistical builder) ---
        for bbox in mask_coords_inner.keys():

            if filter_edge_one_side(edge_x, edge_y, image_shape, bbox):
                continue

            # inner ROI
            img_i, _ = build_cropped_interpolated_roi(
                image_rgb=image_rgb,
                coords=mask_coords_inner[bbox],
                bbox=bbox,
                target_size=self.target_size,
            )

            # outer ROI
            img_o, _ = build_cropped_interpolated_roi(
                image_rgb=image_rgb,
                coords=mask_coords_outer[bbox],
                bbox=bbox,
                target_size=self.target_size,
            )

            inner_imgs.append(np.transpose(img_i / 255.0, (2, 0, 1)))
            outer_imgs.append(np.transpose(img_o / 255.0, (2, 0, 1)))

            geo = geos_all[bbox]
            geo_mat.append([geo["area"], geo["perimeter"], geo["aspect_ratio"]])

            idx = len(valid_bboxes)
            identifiers[idx] = (frame_ID, bbox)
            valid_bboxes.append(bbox)

        if len(valid_bboxes) == 0:
            raise RuntimeError("No valid droplets after edge filtering.")

        # --- stack tensors ---
        inner_tensor = torch.from_numpy(np.stack(inner_imgs)).float()
        outer_tensor = torch.from_numpy(np.stack(outer_imgs)).float()

        # --- AE encoding ---
        inner_latent = self._encode_batch(self.model_inner, inner_tensor)
        outer_latent = self._encode_batch(self.model_outer, outer_tensor)

        out = {
            "identifiers": identifiers,
            "inner_color_features": inner_latent.astype(np.float32),
            "outer_color_features": outer_latent.astype(np.float32),
            "geo_features": np.asarray(geo_mat, dtype=np.float32),
        }

        if save:
            os.makedirs(save_dir, exist_ok=True)
            save_pickle(out, os.path.join(save_dir, f"{frame_ID}_ae.pkl"))
        else:
            return out

        # --- cleanup ---
        del image_rgb_raw, image_rgb
        del inner_imgs, outer_imgs, inner_tensor, outer_tensor
        gc.collect()


class AEFeaturePipeline:
    def __init__(
        self,
        latent_dim_inner: int,
        latent_dim_outer: int,
        target_size=(32, 32),
        batch_size: int = 32,
        num_epochs: int = 800,
        lr: float = 1e-4,
    ):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("PyTorch version:", torch.__version__)
        print("CUDA is available:", torch.cuda.is_available())
        print("Number of GPUs available:", torch.cuda.device_count())
        print(torch.cuda.get_device_name()) if torch.cuda.is_available() else print("No cuda available.")

        self.device = torch.device(device)
        self.latent_dim_inner = latent_dim_inner
        self.latent_dim_outer = latent_dim_outer
        self.target_size = target_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr

    # INTERNAL: dataset construction
    def _collect_rois(
        self,
        image_rgb_raw_all,
        image_pos_ref,
        image_bri_ref,
        mask_coords,
    ):
        imgs = []

        for image_rgb_raw in image_rgb_raw_all:
            image_rgb = image_preprocessing(
                image_rgb_raw, image_pos_ref, image_bri_ref
            )

            for bbox, coords in mask_coords.items():
                roi, _ = build_cropped_interpolated_roi(
                    image_rgb=image_rgb,
                    coords=coords,
                    bbox=bbox,
                    target_size=self.target_size,
                )
                roi = np.transpose(roi / 255.0, (2, 0, 1))
                imgs.append(roi)

        X = torch.tensor(np.stack(imgs), dtype=torch.float32)
        mask = torch.ones((X.size(0), 1, X.size(2), X.size(3)))
        return X, mask

    # INTERNAL: train one AE
    def _train_single_ae(self, X, mask, latent_dim, model_path):
        model = AEConvFC(latent_dim=latent_dim).to(self.device)

        loader = DataLoader(
            TensorDataset(X, mask),
            batch_size=self.batch_size,
            shuffle=True,
        )

        train_ae_conv_fc(
            model=model,
            train_loader=loader,
            val_loader=loader,
            device=self.device,
            num_epochs=self.num_epochs,
            lr=self.lr,
            model_path=model_path,
        )

        model.eval()
        return model

    # PUBLIC: run everything
    def build_features(
        self,
        image_rgb_raw_all,
        frame_idx_all: List[int],
        image_pos_ref,
        image_bri_ref,
        mask_coords_inner: Dict,
        mask_coords_outer: Dict,
        geos_all: Dict,
        save_dir: str,
    ):
        """
        Train AE models (if needed) and generate AE features for all frames.
        """

        os.makedirs(save_dir, exist_ok=True)

        # -------------------------
        # 1. Train INNER AE
        # -------------------------
        print("Training AE for INNER ROIs...")
        X_inner, mask_inner = self._collect_rois(
            image_rgb_raw_all,
            image_pos_ref,
            image_bri_ref,
            mask_coords_inner,
        )

        ae_inner = self._train_single_ae(
            X_inner,
            mask_inner,
            self.latent_dim_inner,
            os.path.join(save_dir, "ae_inner.pth"),
        )

        # -------------------------
        # 2. Train OUTER AE
        # -------------------------
        print("Training AE for OUTER ROIs...")
        X_outer, mask_outer = self._collect_rois(
            image_rgb_raw_all,
            image_pos_ref,
            image_bri_ref,
            mask_coords_outer,
        )

        ae_outer = self._train_single_ae(
            X_outer,
            mask_outer,
            self.latent_dim_outer,
            os.path.join(save_dir, "ae_outer.pth"),
        )

        # -------------------------
        # 3. Feature extraction
        # -------------------------
        builder = AEDualROIFeatureBuilder(
            model_inner=ae_inner,
            model_outer=ae_outer,
            device=self.device,
        )

        for image_rgb_raw, frame_ID in zip(image_rgb_raw_all, frame_idx_all):
            print(f"Extracting AE features for frame {frame_ID}...")

            builder.build_per_frame_features(
                image_rgb_raw=image_rgb_raw,
                frame_ID=frame_ID,
                image_pos_ref=image_pos_ref,
                image_bri_ref=image_bri_ref,
                mask_coords_inner=mask_coords_inner,
                mask_coords_outer=mask_coords_outer,
                geos_all=geos_all,
                save=True,
                save_dir=save_dir,
            )

        print("AE feature extraction complete.")