# Direct copy from: https://discuss.pytorch.org/t/pytorch-lightning-for-prediction/128403/4

import matplotlib
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from pytorch_lightning import Trainer, seed_everything, LightningDataModule, LightningModule

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class SeqDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.data = torch.load("traindata.pt")

    def train_dataloader(self):
        input = torch.from_numpy(self.data[3:, :-1])
        target = torch.from_numpy(self.data[3:, 1:])
        dataset = TensorDataset(input, target)
        print(input.shape, target.shape)
        return DataLoader(dataset, batch_size=len(input))

    def val_dataloader(self):
        input = torch.from_numpy(self.data[:3, :-1])
        target = torch.from_numpy(self.data[:3, 1:])
        dataset = TensorDataset(input, target)
        return DataLoader(dataset, batch_size=len(input))


class Sequence(LightningModule):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):  # if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def configure_optimizers(self):
        return optim.LBFGS(self.parameters(), lr=0.8)

    def training_step(self, batch, batch_idx):
        input, target = batch
        out = self(input)
        loss = F.mse_loss(out, target)
        self.print("loss:", loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        future = 1000
        pred = self(input, future=future)
        loss = F.mse_loss(pred[:, :-future], target)
        print("test loss:", loss.item())
        y = pred.detach().numpy()
        # draw the result

        plt.figure(figsize=(30, 10))
        plt.title("Predict future values for time sequences\n(Dashlines are predicted values)", fontsize=30)
        plt.xlabel("x", fontsize=20)
        plt.ylabel("y", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[: input.size(1)], color, linewidth=2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1) :], color + ":", linewidth=2.0)

        draw(y[0], "r")
        draw(y[1], "g")
        draw(y[2], "b")
        plt.savefig(f"predict{self.global_step:d}.pdf")
        plt.close()


if __name__ == "__main__":

    import numpy as np
    import torch

    np.random.seed(2)

    T = 20
    L = 1000
    N = 100

    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float64')
    torch.save(data, open('traindata.pt', 'wb'))
    
    seed_everything(0)
    trainer = Trainer(max_steps=15, precision=64)
    model = Sequence()
    datamodule = SeqDataModule()
    trainer.fit(model, datamodule)