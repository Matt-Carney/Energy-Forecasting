### Import libraries
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

#os.chdir("../../..")

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from lightning.pytorch.tuner import Tuner
import matplotlib.pyplot as plt
import pandas as pd
import torch

import pytorch_forecasting as ptf
from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import MAE, RMSE, MAPE, MultivariateNormalDistributionLoss
from pytorch_forecasting.data import TorchNormalizer
### DeepAR with for 2018 ###

### Import data
data = Path('data')
fp_2018 = data/'CAISO_zone_1_2018.csv'
df_2018 = pd.read_csv(fp_2018)

### Train, valid, test split
#val_start = '2018-09-01 00:00:00'
#val_end = '2018-12-01 00:00:00'
df_2018['holiday'] = df_2018['holiday'].astype(str)

#df_2018['DHI_lag'] = df_2018['DHI'].shift(1)
#df_2018['Temperature_lag'] = df_2018['Temperature'].shift(1)

#df_2018= df_2018.iloc[1:,:]
df_2018.columns
df_2018['hour'] = pd.to_datetime(df_2018['date_time']).dt.hour

df_2018['series'] = 'A' 

train_cutoff_idx = df_2018[df_2018['date_time'] == '2018-09-30 23:00:00']['time_idx'].values[0] # returns index of training cutoff
val_cutoff_idx = df_2018[df_2018['date_time'] == '2018-11-30 23:00:00']['time_idx'].values[0] # returns index of validation cutoff


#lag_dict = {"DHI" : [1],
#            "DNI": [1],
#            "Dew Point": [1],
#            "Solar Zenith Angle": [1],
#            "Wind Speed": [1],
#            "Relative Humidity": [1],
#            "Temperature": [1]}


max_encoder = 48
max_pred = 24
train_cutoff = train_cutoff_idx - max_pred

training = TimeSeriesDataSet(
    df_2018[lambda x: x.time_idx <= train_cutoff], #seems more pandas/pythonic but leaving as is for now
    time_idx="time_idx",
    target="load_power",
    group_ids=['series'],
    time_varying_unknown_reals=["load_power"],
    time_varying_known_reals=[#'month_day',
                              "DNI_lag",
                              "Dew Point_lag",
                              "Solar Zenith Angle_lag",
                              "Wind Speed_lag",
                              "Relative Humidity_lag",
                              "Temperature_lag", 'hour'],
    time_varying_known_categoricals= ['holiday'],
    #lags= lag_dict,
    max_encoder_length=max_encoder,
    max_prediction_length=max_pred,
    target_normalizer=TorchNormalizer(method='identity', center=True, transformation=None, method_kwargs={}), # https://github.com/jdb78/pytorch-forecasting/issues/1220
    add_target_scales=True,)

validation = TimeSeriesDataSet.from_dataset(training, df_2018[lambda x: x.time_idx<=val_cutoff_idx], min_prediction_idx=train_cutoff_idx + 1)
testing = TimeSeriesDataSet.from_dataset(training, df_2018, min_prediction_idx=val_cutoff_idx - 1)

batch_size = 128
# synchronize samples in each batch over time - only necessary for DeepVAR, not for DeepAR
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0, batch_sampler="synchronized")
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size*10, num_workers=0, batch_sampler="synchronized")
test_dataloader = testing.to_dataloader(train=False, batch_size=batch_size*10, num_workers=0, batch_sampler="synchronized")

val_actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
test_actuals = torch.cat([y[0] for x, y in iter(test_dataloader)])

# Baseline estimator and absolute error on validation
#baseline_predictions = Baseline().predict(val_dataloader, trainer_kwargs=dict(accelerator="cpu"), return_y=True)
#actuals = torch.cat([y[0] for x, y in iter(val_dataloader)]) # baseline_predictions.y[0].shape using baseline_predictions.y requires a reshape
#baseline_predictions.shape
#actuals.shape
#RMSE()(baseline_predictions, actuals)

# Baseline estimator and absolute error on test
##aseline_predictions2 = Baseline().predict(test_dataloader, trainer_kwargs=dict(accelerator="cpu"))
#actuals2 = torch.cat([y[0] for x, y in iter(test_dataloader)]) 
#baseline_predictions2.shape
#actuals2.shape
#SMAPE()(baseline_predictions2, actuals2)


# Initiate model 
pl.seed_everything(42)
trainer = pl.Trainer(accelerator="cpu", gradient_clip_val=1e-1)
net = DeepAR.from_dataset( # Look at create log
    training,
    learning_rate=3e-2,
    hidden_size=30,
    rnn_layers=2,
    loss=MultivariateNormalDistributionLoss(rank=30),
    optimizer="Adam")


# Find optimal learning rate # Error with this and mps https://github.com/pytorch/pytorch/issues/98074, ran on cpu and optimal learning rate is 0.0501188, switching back to mps
res = Tuner(trainer).lr_find(
    net,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    min_lr=1e-5,
    max_lr=1e0,
    early_stop_threshold=100)

print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
#fig.show()
print(f'Suggested learning rate {res.suggestion()}')
net.hparams.learning_rate = res.suggestion()


### Training
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
tensorboard = pl_loggers.TensorBoardLogger('./')
trainer = pl.Trainer(
    max_epochs=20,
    accelerator="cpu",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback],
    limit_train_batches=50,
    limit_val_batches=50,
    enable_checkpointing=True,
    logger=tensorboard
)


net = DeepAR.from_dataset(
    training,
    learning_rate=1e-2,
    log_interval=5,
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


# Fit model
best_model_path = trainer.checkpoint_callback.best_model_path
best_model = DeepAR.load_from_checkpoint(best_model_path)

# Predictions
val_predictions = best_model.predict(val_dataloader, trainer_kwargs=dict(accelerator="cpu"))
RMSE()(val_predictions, val_actuals)
MAE()(val_predictions, val_actuals)

test_predictions = best_model.predict(test_dataloader, trainer_kwargs=dict(accelerator="cpu"))
RMSE()(test_predictions, test_actuals)
MAE()(test_predictions, test_actuals)
MAPE()(test_predictions, test_actuals)

#predictions.output
#predictions.y

#raw_predictions = net.predict(val_dataloader, mode="raw", return_x=True, n_samples=100, trainer_kwargs=dict(accelerator="cpu"))

# Plotting
#series = validation.x_to_index(raw_predictions.x)["series"]
#for idx in range(10):  # plot 10 examples
#    best_model.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True)
#    plt.suptitle(f"Series: {series.iloc[idx]}")
#    plt.show()
#best_model.plot_prediction(raw_predictions.x, raw_predictions.output, idx=0, add_loss_to_title=True)
#plt.show()
#plt.close()


#predictions2 = best_model.predict(test_dataloader, trainer_kwargs=dict(accelerator="cpu"))
#RMSE()(predictions2, actuals2)