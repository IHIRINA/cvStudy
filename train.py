from model.plmodel import MyPlModel, SaveCheck
from utils.dataset import MriForGAN_once, dict_as_namespace
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import CSVLogger
import pytorch_lightning as pl
import yaml

with open('./options.yaml', 'r') as f:
    options = yaml.load(f, Loader=yaml.FullLoader)

def train(options):
    # load options
    options = dict_as_namespace(options)
    root = options.data.hdf5_root

    # load dataset
    train_dataset = MriForGAN_once('train', options.data.use_slice, root, options.data.use_modality)
    val_dataset = MriForGAN_once('valid', options.data.use_slice, root, options.data.use_modality)
    train_dataloader = DataLoader(train_dataset, batch_size=options.train.batch_size, shuffle=True, num_workers=15)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=15)
    l_train = len(train_dataloader)
    l_val = len(val_dataloader)

    # others
    SaveCalls = [SaveCheck(options)]
    logger = CSVLogger('/root/autodl-tmp/logs', name='loss', flush_logs_every_n_steps=1000)

    # load model
    model_pl = MyPlModel(options, l_train, l_val)
    trainer = pl.Trainer(
        accumulate_grad_batches=1, 
        accelerator='gpu',
        devices=options.train.cuda_num, 
        max_epochs=options.train.epochs,
        precision=32,
        callbacks=SaveCalls,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        logger=logger,
        log_every_n_steps=5,
        enable_checkpointing=False,
        # strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=True),
    )

    if options.train.start != 0:
        trainer.fit(
            model=model_pl, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader,
            ckpt_path=f"/root/autodl-tmp/logs/checkpoints/epoch_{options.train.start}.ckpt"
        )
    else:
        trainer.fit(
            model=model_pl, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader,
        )



if __name__ == '__main__':
    train(options)
