# Ref: https://discuss.pytorch.org/t/pytorch-lightning-for-prediction/128403/4
# Ref: https://github.com/pytorch/examples/tree/master/time_sequence_prediction

from pytorch_lightning.utilities.seed import seed_everything
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader

from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_FILENAME = "sine_traindata.pt"


class SeqDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.data = torch.load(DATA_FILENAME)

    def train_dataloader(self):
        inputs = torch.from_numpy(self.data[3:, :-1])
        targets = torch.from_numpy(self.data[3:, 1:])
        dataset = TensorDataset(inputs, targets)
        print(inputs.shape, targets.shape)
        return DataLoader(dataset, batch_size=len(inputs))

    def val_dataloader(self):
        inputs = torch.from_numpy(self.data[:3, :-1])
        targets = torch.from_numpy(self.data[:3, 1:])
        dataset = TensorDataset(inputs, targets)
        return DataLoader(dataset, batch_size=len(inputs))


class Sequence(LightningModule):
    def __init__(self, in_size, h_size, out_size, learn_rate):
        '''Define model architecture (e.g. layers)
        Args:
            in_size : int
                size of model inputs
            h_size : int
                size of hidden layers
            out_size : int
                size of model outpu
        '''
        super().__init__()
        self.h_size = h_size
        self.learn_rate = learn_rate
        self.lstm1 = nn.LSTMCell(in_size, h_size)
        self.lstm2 = nn.LSTMCell(h_size, h_size)
        self.linear = nn.Linear(h_size,out_size)

    def forward(self, inputs, future=0):
        outputs = []
        h_t = torch.zeros(inputs.size(0), self.h_size, dtype=torch.double)
        c_t = torch.zeros(inputs.size(0), self.h_size, dtype=torch.double)
        h_t2 = torch.zeros(inputs.size(0), self.h_size, dtype=torch.double)
        c_t2 = torch.zeros(inputs.size(0), self.h_size, dtype=torch.double)

        for input_t in inputs.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        for i in range(future): 
            # predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def configure_optimizers(self):
        return optim.LBFGS(self.parameters(), lr=self.learn_rate)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.mse_loss(outputs, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        future = 1000
        pred = self(inputs, future=future)
        loss = F.mse_loss(pred[:, :-future], targets)
        y = pred.detach().numpy()

        # draw the result
        plt.figure(figsize=(30, 10))
        plt.title("Predict future values for time sequences\n(Dashlines are predicted values)", fontsize=30)
        plt.xlabel("x", fontsize=20)
        plt.ylabel("y", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        def draw(yi, color):
            plt.plot(np.arange(inputs.size(1)), yi[: inputs.size(1)], color, linewidth=2.0)
            plt.plot(np.arange(inputs.size(1), inputs.size(1) + future), yi[inputs.size(1) :], color + ":", linewidth=2.0)

        draw(y[0], "r")
        draw(y[1], "g")
        draw(y[2], "b")
        plt.savefig(f"predict{self.global_step:d}.pdf")
        plt.close()

if __name__ == "__main__":

    # generate and save data
    np.random.seed(2)

    T = 20
    L = 1000
    N = 100

    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float64')
    torch.save(data, open(DATA_FILENAME, 'wb'))

    seed_everything(0)
    trainer = Trainer(max_steps=15, precision=64)
    model = Sequence(1, 51, 1, 0.8)
    datamodule = SeqDataModule()
    trainer.fit(model, datamodule)


    

