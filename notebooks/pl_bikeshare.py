# Ref: https://github.com/shotleft/how-to-python/blob/master/How%20it%20works%20-%20Bike%20Share%20Regression%20PyTorch%20Lightning.ipynb

# Let's import some basic libraries
import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# imports for model definition and training
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.profiler import AdvancedProfiler

L_RATE = 0.2
MSE_LOSS = nn.MSELoss(reduction='mean')
MAX_EPOCHS= 50

class RegressionModule(pl.LightningModule):
    ''' The Model'''
    def __init__(self, n_features, n_targets):
        ''' Establish model architecture (i.e. layers)
        '''
        super().__init__()

        # Here we have one input layer (size 56 as we have 56 features), one hidden layer (size 10), 
        # and one output layer (size 1 as we are predicting a single value)
        self.fc1 = nn.Linear(n_features, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        ''' Define forward pass through model and outputs

        NOTE: Is this really the best way to define forward pass? Can you also define activation functions
        in __init__ and then have forward be a trivial implementation?

        NOTE: do you really want to modify input x in place as is done in the example?
        '''

        # We're using the sigmoid activation function on our hidden layer, but our output layer has no activation 
        # function as we're predicting a continuous variable so we want the actual number predicted
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

    def configure_optimizers(self):
        ''' Define optimizaer to use'''
        # NOTE: I'm changing example from SGD to ADAM
        return optim.Adam(self.parameters(), lr=L_RATE)

    def training_step(self, batch, batch_idx):
        ''' Process to run in training step'''
        # NOTE: example uses the term "logits", but this seems like a severe abuse of the term
        # given that this is a regression problem, not a classification problem
        x, y = batch
        y_out = self.forward(x)
        loss = MSE_LOSS(y_out, y)
        # logs = {'loss': loss}
        # return {'loss': loss, 'log': logs}
        self.log('train_loss', loss, prog_bar=True)
        # logs = {'train_loss': loss}
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_train_loss', avg_loss, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        ''' Process to run for validation '''
        x, y = batch
        y_out = self.forward(x)
        loss = MSE_LOSS(y_out, y)
        # self.log('validation_loss', loss)
        return {'validation_loss': loss}
        # return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['validation_loss'] for x in outputs]).mean()
        # tensorboard_logs = {'validation_loss': avg_loss}
        # return {'avg_validation_loss': avg_loss, 'log': tensorboard_logs}
        self.log('avg_validation_loss', avg_loss, prog_bar=True)
        return avg_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_out = self.forward(x)
        loss = MSE_LOSS(y_out, y)
        # self.log('test_loss', loss)
        return {'test_loss': loss}
        # return loss

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        # logs = {'test_loss': avg_loss}      
        self.log('avg_test_loss', avg_loss,prog_bar=True)
        # return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs }
        return {'avg_test_loss': avg_loss}
        # return avg_loss



def bikeshare_dataloader(features, targets, batch_size):
    '''Define how to load data into model

    NOTE: there was an auto-completed version of this code that simply called super().train_dataloader().
    Should this be used instead as the more general approach or is the below modifications important?

    NOTE: is it really the best practice to hard code the batch size here in this function?
    
    NOTE: in pytorch lightning's own docs they seem to separate out train, val, test, and predict
    dataloaders quite explicitly (and somewhat redundantly) but also use a different class
    for the entire data module: 
    https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html#lightningdatamodules
    '''
    # auto generated version of this function
    # return super().train_dataloader()

    dataset = TensorDataset(
        torch.tensor(features.values).float(), 
        torch.tensor(targets[['cnt']].values).float()
    )
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size)
    return train_loader


if  __name__ == "__main__":

    # import and view the data
    curdir = pathlib.Path(__file__).parent.resolve()
    df = pd.read_csv(curdir.joinpath('bike_sharing_hourly.csv'))
    print(df.head())

    # Being a timeseries problem, let's look at the date ranges we have available
    # Oooo...First time I'm learning about F-strings: 
    # https://realpython.com/python-f-strings/
    # interesting behavior, they even generate the line returns and tabs!
    print(f"""
    Earliest date - {df['dteday'].min()} 
    Latest date - {df['dteday'].max()}
    Total number of days - {len(df) / 24}
    """)
    # visualize data
    # df[df['dteday'].isin(df['dteday'].unique()[0:14])].plot(x='dteday', y='cnt', figsize=(16, 4))
    # plt.show()

    # one-hot encode categorical variables
    onehot_fields = ['season', 'mnth', 'hr', 'weekday', 'weathersit']
    for field in onehot_fields:
        dummies = pd.get_dummies(df[field], prefix=field, drop_first=False)
        df = pd.concat([df, dummies], axis=1)

    # backup and remove data we don't use
    df_backup = df.copy()
    df.drop(onehot_fields, axis=1, inplace=True)
    print(df.head())

    # normalize numerical data to mean 0, std dev=1
    # store scalings in a dictionary for future reference
    numerical_fields = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
    scaled_features = {}
    for field in numerical_fields:
        mean, std = df[field].mean(), df[field].std()
        scaled_features[field] = [mean, std]
        df.loc[:, field] = (df[field] - mean)/std
    print(scaled_features)

    # remove further unused fields
    fields_to_drop = ['instant', 'dteday', 'atemp', 'workingday']
    df.drop(fields_to_drop, axis=1, inplace=True)
    print(df.head())

    # Split up the training, validation, and test datasets
    # Since it is timeserires, it makes sense to train on earlier data
    # and validate/test on later data
    # NOTE: I am breaking up the data differently to better 
    # disentangle training, validation, and test
    n_data = len(df)
    n_train_data = int(0.90*n_data)+1
    n_validation_data = int(0.05*n_data)+1
    n_test_data = n_data - (n_train_data + n_validation_data) 
    train_data = df[:n_train_data]
    validation_data = df[n_train_data:n_train_data+n_validation_data]
    test_data = df[-n_test_data:]
    print(f"""
            {n_train_data}   train
        +   {n_validation_data}    validation 
        +   {n_test_data}    test 
        ----------------------------------
            {n_train_data + n_validation_data + n_test_data}
        ----------------------------------
            {n_data}            total""")

    # Separate data into feature and target fields
    target_fields = ['cnt', 'casual', 'registered']
    train_features, train_targets = train_data.drop(target_fields, axis=1), train_data[target_fields]
    validation_features, validation_targets = validation_data.drop(target_fields, axis=1), validation_data[target_fields]
    test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
    n_features = train_features.shape[1]
    n_targets = 1       # not sure why casual and registered are target fields
    assert validation_features.shape[1] == test_features.shape[1] == n_features

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # DEFINE MODEL
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # instantiate regression module
    regmod = RegressionModule(n_features=n_features, n_targets=n_targets)

    # instantiate trainer dataloader
    train_loader = bikeshare_dataloader(train_features, train_targets, batch_size=128)

    # instantiate trainer
    trainer = Trainer(max_epochs = MAX_EPOCHS)

    # train module
    trainer.fit(regmod, train_loader)

    # validate
    validation_loader = bikeshare_dataloader(validation_features, validation_targets, batch_size=128)
    trainer.validate(dataloaders=validation_loader)

    # test
    test_loader = bikeshare_dataloader(test_features, test_targets, batch_size=128)
    trainer.test(test_dataloaders=test_loader)







