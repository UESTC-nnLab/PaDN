"""Shared sequence dataset used by all three training entry points.

The original project carried four nearly identical loaders with dataset-specific
absolute paths.  This module keeps paths in the training configuration and makes
the frame/annotation handling identical across datasets.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _read_annotations(path: Path):
    samples = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            fields = raw_line.strip().split()
            if not fields:
                continue
            boxes = []
            for raw_box in fields[1:]:
                values = raw_box.split(",")
                if len(values) != 5:
                    raise ValueError(
                        f"{path}:{line_number}: expected x1,y1,x2,y2,class, got {raw_box!r}"
                    )
                boxes.append([float(value) for value in values])
            box_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
            samples.append((str(Path(fields[0]).expanduser()), box_array))
    if not samples:
        raise ValueError(f"annotation file is empty: {path}")
    return samples


def _previous_frame(path: str, offset: int) -> str:
    frame = Path(path)
    try:
        index = int(frame.stem)
    except ValueError as exc:
        raise ValueError(f"frame filename must have a numeric stem: {path}") from exc
    previous = max(index - offset, 0)
    stem = str(previous).zfill(len(frame.stem)) if frame.stem.startswith("0") else str(previous)
    return str(frame.with_name(stem + frame.suffix))


def _resize_image_and_boxes(image_path: str, boxes: np.ndarray, size: int):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        source_w, source_h = image.size
        scale = min(size / source_w, size / source_h)
        resized_w, resized_h = int(source_w * scale), int(source_h * scale)
        offset_x, offset_y = (size - resized_w) // 2, (size - resized_h) // 2
        image = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (size, size), (128, 128, 128))
        canvas.paste(image, (offset_x, offset_y))

    scaled = boxes.copy()
    if len(scaled):
        scaled[:, [0, 2]] = scaled[:, [0, 2]] * resized_w / source_w + offset_x
        scaled[:, [1, 3]] = scaled[:, [1, 3]] * resized_h / source_h + offset_y
        scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0, size)
        scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0, size)
        valid = (scaled[:, 2] - scaled[:, 0] > 1) & (scaled[:, 3] - scaled[:, 1] > 1)
        scaled = scaled[valid]
    return np.asarray(canvas, dtype=np.float32), scaled


def _xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    boxes = boxes.copy()
    if len(boxes):
        boxes[:, 2:4] -= boxes[:, 0:2]
        boxes[:, 0:2] += boxes[:, 2:4] / 2
    return boxes


class SequenceDataset(Dataset):
    def __init__(
        self,
        annotation_path: str,
        image_size: int = 512,
        num_frames: int = 2,
        training: bool = True,
        embedding_path: Optional[str] = None,
        relation_path: Optional[str] = None,
    ):
        self.annotation_path = Path(annotation_path).expanduser().resolve()
        if not self.annotation_path.is_file():
            raise FileNotFoundError(f"annotation file not found: {self.annotation_path}")
        self.image_size = image_size
        self.num_frames = num_frames
        self.training = training
        self.samples = _read_annotations(self.annotation_path)
        self.boxes_by_path = {path: boxes for path, boxes in self.samples}

        self.embeddings = self._load_pickle(embedding_path, "embedding")
        self.relations = self._load_pickle(relation_path, "relation")
        if self.training and ((self.embeddings is None) != (self.relations is None)):
            raise ValueError("embedding_path and relation_path must be provided together")

        # Fail before spawning workers instead of surfacing a cryptic PIL error later.
        missing = [path for path, _ in self.samples[:100] if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"image referenced by annotations was not found: {missing[0]}")

    @staticmethod
    def _load_pickle(path: Optional[str], kind: str):
        if not path:
            return None
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"{kind} file not found: {resolved}")
        with resolved.open("rb") as handle:
            value = pickle.load(handle)
        if not isinstance(value, dict):
            raise TypeError(f"{kind} file must contain a dictionary: {resolved}")
        return value

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        current_path, current_boxes = self.samples[index]
        frame_paths = [
            _previous_frame(current_path, offset)
            for offset in reversed(range(self.num_frames))
        ]

        images = []
        multi_boxes = []
        for frame_path in frame_paths:
            if not Path(frame_path).is_file():
                frame_path = current_path
            frame_boxes = self.boxes_by_path.get(frame_path, current_boxes)
            image, scaled_boxes = _resize_image_and_boxes(
                frame_path, frame_boxes, self.image_size
            )
            images.append(image)
            multi_boxes.append(_xyxy_to_cxcywh(scaled_boxes))

        # Targets correspond to the current (last) frame.
        target = multi_boxes[-1].copy()
        images = np.stack(images, axis=0)
        images = (images / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        images = np.transpose(images, (3, 0, 1, 2)).astype(np.float32)

        captions = None
        relation = None
        if self.training and self.embeddings is not None:
            try:
                captions = np.stack([self.embeddings[path] for path in frame_paths])
                relation = np.asarray([self.relations[current_path]])
            except KeyError as exc:
                raise KeyError(
                    f"motion metadata has no entry for {exc.args[0]!r}; "
                    "annotation and pickle paths must use the same image names"
                ) from exc

        return images, target, captions, multi_boxes, relation


def sequence_collate(batch):
    images, targets, captions, multi_targets, relations = zip(*batch)
    return (
        torch.from_numpy(np.stack(images)).float(),
        [torch.from_numpy(target).float() for target in targets],
        list(captions),
        list(multi_targets),
        list(relations),
    )
