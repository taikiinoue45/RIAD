from pathlib import Path
from typing import Dict, List

import cv2
import pandas as pd
from riad.albu import Compose
from torch.utils.data import Dataset


class MVTecDataset(Dataset):
    def __init__(self, root: str, query_list: List[str], augs_dict: Dict[str, Compose]) -> None:

        self.root = Path(root)
        self.augs_dict = augs_dict
        self.stem_list = []

        df = pd.read_csv(self.root / "info.csv")
        for q in query_list:
            stem = df.query(q)["stem"]
            self.stem_list += stem.to_list()

    def __getitem__(self, idx: int) -> dict:

        stem = self.stem_list[idx]
        img = cv2.imread(str(self.root / f"images/{stem}.png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.root / f"masks/{stem}.png"))

        raw_img = self.augs_dict["preprocess"](image=img)["image"]
        raw_img = self.augs_dict["totensor"](image=raw_img)["image"]

        data_dict = {"image": img, "mask": mask}
        data_dict = self.augs_dict["preprocess"](**data_dict)
        data_dict = self.augs_dict["cutout"](**data_dict)
        data_dict = self.augs_dict["totensor"](**data_dict)

        data_dict["raw_image"] = raw_img
        data_dict["stem"] = stem
        return data_dict

    def __len__(self) -> int:

        return len(self.stem_list)
