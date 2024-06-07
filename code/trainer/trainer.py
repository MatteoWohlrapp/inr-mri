import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from utils.visualization import show_batch, show_image
import sys
import os
from utils.tiling import extract_with_inner_patches, reconstruct_image_from_inner_patches
import matplotlib.pyplot as plt


def create_tqdm_bar(iterable, desc, limit_io=False):
    if not limit_io:
        return tqdm(
            enumerate(iterable),
            total=len(iterable),
            ncols=150,
            desc=desc,
            file=sys.stdout,
        )
    else:
        return enumerate(iterable)


class Trainer:
    def __init__(
        self,
        model,
        device,
        train_dataset,
        optimizer,
        output_dir,
        val_dataset=None,
        batch_size=1,
        output_name="output",
        scheduler=None,
        limit_io=False,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=True
            )
        self.criterion = nn.MSELoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = SummaryWriter(log_dir=f"{output_dir}/runs/{output_name}")
        self.output_name = output_name
        self.output_dir = output_dir
        self.limit_io = limit_io

    def train(self, num_epochs):
        validation_loss = 0
        for epoch in range(num_epochs):
            self.model.train()
            training_loop = create_tqdm_bar(
                self.train_loader,
                desc=f"Training Epoch [{epoch}/{num_epochs}]",
                limit_io=self.limit_io,
            )
            training_loss = 0
            max_train_iteration = 0
            max_val_iteration = 0
            for train_iteration, (undersampled_batch, fully_sampled_batch) in training_loop:
                undersampled_batch = undersampled_batch.to(self.device)
                fully_sampled_batch = fully_sampled_batch.to(self.device)

                undersampled_patches, _ = extract_with_inner_patches(undersampled_batch, 32, 16)
                fully_sampled_patches, _ = extract_with_inner_patches(fully_sampled_batch, 32, 16)

                self.optimizer.zero_grad()
                outputs = self.model(undersampled_patches)
                loss = self.criterion(outputs, fully_sampled_patches)
                loss.backward()
                self.optimizer.step()

                training_loss += loss.item()
                max_train_iteration = train_iteration

                if not self.limit_io:
                    # Update the progress bar.
                    training_loop.set_postfix(
                        train_loss="{:.8f}".format(
                            training_loss / (train_iteration + 1)
                        ),
                        val_loss="{:.8f}".format(validation_loss),
                    )
                    training_loop.refresh()

                    # Update the tensorboard logger.
                    self.writer.add_scalar(
                        "Training Loss",
                        loss.item(),
                        epoch * len(self.train_loader) + train_iteration,
                    )

            # Validation
            if self.val_dataset is not None:
                self.model.eval()
                val_loop = create_tqdm_bar(
                    self.val_loader, desc=f"Validation Epoch [{epoch}/{num_epochs}]"
                )
                validation_loss = 0
                with torch.no_grad():
                    for val_iteration, img_batch in val_loop:
                        img_batch = img_batch.to(self.device)
                        outputs = self.model(img_batch)
                        loss = self.criterion(outputs, img_batch)
                        validation_loss += loss.item()

                        if not self.limit_io:
                            # Update the progress bar.
                            val_loop.set_postfix(
                                val_loss="{:.8f}".format(
                                    validation_loss / (val_iteration + 1)
                                )
                            )
                            val_loop.refresh()

                            # Update the tensorboard logger.
                            self.writer.add_scalar(
                                f"Validation Loss",
                                validation_loss / (val_iteration + 1),
                                epoch * len(self.val_loader) + val_iteration,
                            )
                        max_val_iteration = val_iteration
            if self.scheduler:
                self.scheduler.step()

            if self.limit_io:
                if self.val_dataset:
                    print(
                        f"Epoch: {epoch}, training loss {training_loss / (max_train_iteration) + 1}, validation loss {validation_loss / (max_val_iteration + 1)}", flush=True
                    )
                else:
                    print(
                        f"Epoch: {epoch}, training loss {training_loss / (max_train_iteration) + 1}", flush=True
                    )

        self.writer.close()

        with torch.no_grad():
            if not os.path.exists(f"{self.output_dir}/model_checkpoints"):
                os.makedirs(f"{self.output_dir}/model_checkpoints")

            torch.save(
                self.model.state_dict(),
                f"{self.output_dir}/model_checkpoints/{self.output_name}_model.pth",
            )
