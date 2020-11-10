from omegaconf import DictConfig
from torch.nn import Module
from tqdm import tqdm


class RunnerTrainValTest:

    model: Module
    cfg: DictConfig

    def run_train(self):

        self.model.train()
        pbar = tqdm(range(1, self.cfg.run.train.epochs + 1), desc="train")
        for epoch in pbar:

            epoch_loss = 0
            for data_dict in self.dataloader_dict["train"]:

                self.optimizer.zero_grad()
                img = data_dict["image"].to(self.cfg.device)
                raw_img = data_dict["raw_image"].to(self.cfg.device)

                reconstructed_img = self.model(img)
                loss_L2 = self.criterion_L2(reconstructed_img, raw_img)
                loss_SSIM = self.criterion_SSIM(reconstructed_img, raw_img)
                loss_MSGM = self.criterion_MSGM(reconstructed_img, raw_img)
                loss = loss_L2 + loss_SSIM + loss_MSGM
                epoch_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss = epoch_loss / len(self.dataloader_dict["train"])
