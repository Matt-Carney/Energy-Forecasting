### Import libraries
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

#os.chdir("../../..")

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.tuner import Tuner
import matplotlib.pyplot as plt
import pandas as pd
import torch

import pytorch_forecasting as ptf
from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import MAE, RMSE, SMAPE, MultivariateNormalDistributionLoss
from pytorch_forecasting.data import TorchNormalizer
### DeepAR with for 2018 ###

### Import data
data = Path('data')
fp_2018 = data/'2018_CAISO_zone_1_.csv'
df_2018 = pd.read_csv(fp_2018)
df_2018 = df_2018[['time_idx', 'load_power']] # Just using time_idx and load_power for intiial run
df_2018['series'] = 0 # Placeholder, need to have a group id, use constant if just have single time series

### Training and validation dataset and dataloaders
# #TODO: Review train/val split and add test data

max_encoder_length = 60
max_prediction_length = 20

training_cutoff = df_2018['time_idx'].max() - max_prediction_length # 524160 - 20 = 524140

context_length = max_encoder_length
prediction_length = max_prediction_length

training = TimeSeriesDataSet(
    df_2018[lambda x: x.time_idx <= training_cutoff], # df_2018.iloc[:training_cutoff+1, :] seems more pandas/pythonic but leaving as is for now
    time_idx="time_idx",
    target="load_power",
    time_varying_unknown_reals=["load_power"],
    group_ids=["series"],
    max_encoder_length=context_length,
    max_prediction_length=prediction_length,
    target_normalizer=TorchNormalizer(method='identity', center=True, transformation=None, method_kwargs={}), # https://github.com/jdb78/pytorch-forecasting/issues/1220
    add_target_scales=True,)

validation = TimeSeriesDataSet.from_dataset(training, df_2018, min_prediction_idx=training_cutoff + 1)
batch_size = 128
# synchronize samples in each batch over time - only necessary for DeepVAR, not for DeepAR
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0, batch_sampler="synchronized")
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized")

# Calculate baseline estimator and absolute error
baseline_predictions = Baseline().predict(val_dataloader, trainer_kwargs=dict(accelerator="mps"), return_y=True)
SMAPE()(baseline_predictions.output, baseline_predictions.y)

# Initiate model 
pl.seed_everything(42)
trainer = pl.Trainer(accelerator="mps", gradient_clip_val=1e-1)
net = DeepAR.from_dataset( # Look at create log
    training,
    learning_rate=3e-2,
    hidden_size=30,
    rnn_layers=2,
    loss=MultivariateNormalDistributionLoss(rank=30),
    optimizer="Adam")


# Find optimal learning rate # Error with this and mps https://github.com/pytorch/pytorch/issues/98074, ran on cpu and optimal learning rate is 0.0501188, switching back to mps
# res = Tuner(trainer).lr_find(
#     net,
#     train_dataloaders=train_dataloader,
#     val_dataloaders=val_dataloader,
#     min_lr=1e-5,
#     max_lr=1e0,
#     early_stop_threshold=100)

# print(f"suggested learning rate: {res.suggestion()}")
# fig = res.plot(show=True, suggest=True)
# #fig.show()
#net.hparams.learning_rate = res.suggestion() # 0.0501188
net.hparams.learning_rate = 0.0501188

### Training
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
trainer = pl.Trainer(
    max_epochs=30,
    accelerator="cpu",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback],
    limit_train_batches=50,
    enable_checkpointing=True,
)


net = DeepAR.from_dataset(
    training,
    learning_rate=1e-2,
    log_interval=10,
    log_val_interval=1,
    hidden_size=30,
    rnn_layers=2,
    optimizer="Adam",
    loss=MultivariateNormalDistributionLoss(rank=30),
)

trainer.fit(
    net,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# ~4 min to run, 13 epochs because of early EarlyStopping

# Fit model
best_model_path = trainer.checkpoint_callback.best_model_path
best_model = DeepAR.load_from_checkpoint(best_model_path)

# Predictions
predictions = best_model.predict(val_dataloader, trainer_kwargs=dict(accelerator="cpu"), return_y=True)
RMSE()(predictions.output, predictions.y) # 0.1655
#predictions.output
#predictions.y

raw_predictions = net.predict(
    val_dataloader, mode="raw", return_x=True, n_samples=100, trainer_kwargs=dict(accelerator="cpu"))

# Plotting
series = validation.x_to_index(raw_predictions.x)["series"]
#for idx in range(10):  # plot 10 examples
#    best_model.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True)
#    plt.suptitle(f"Series: {series.iloc[idx]}")
#    plt.show()
best_model.plot_prediction(raw_predictions.x, raw_predictions.output, idx=0, add_loss_to_title=True)
plt.show()
plt.close()