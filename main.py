import fire
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from configs.common import config
from hawkdet.utils.yacs import CfgNode as CN
from hawkdet.models.build import build_detor
from hawkdet.dataset import WiderFace, detection_collate, preproc
from hawkdet.lib.prior_box import PriorBox
from hawkdet.loss import MultiBoxLoss

from pathlib import Path
from datetime import datetime

import ignite
import ignite.distributed as idist
from ignite.contrib.engines import common
from ignite.contrib.handlers import PiecewiseLinear, LRScheduler
from torch.optim.lr_scheduler import MultiStepLR
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.metrics import Accuracy, Loss
from ignite.utils import manual_seed, setup_logger


def run(
    seed=2021,
    lr=1e-3,
    batch_size=32,
    max_epoch=250,
    num_workers=4,
    output_path='det-outputs',
    log_every_iters=10,
    resume_from=None,
    with_amp=False,
    backend=None,
    nproc_per_node=None,
    **spawn_kwargs):

    # catch all local parameters
    run_config = locals()
    run_config.update(run_config["spawn_kwargs"])
    del run_config["spawn_kwargs"]
    config.merge_from_other_cfg(CN(run_config))
    spawn_kwargs["nproc_per_node"] = nproc_per_node

    with idist.Parallel(backend=backend, **spawn_kwargs) as parallel:
        parallel.run(training, config)


def training(local_rank, cfg):
    rank = idist.get_rank()
    manual_seed(cfg.seed + rank)
    device = idist.device()

    logger = setup_logger(name="HawkDet-Training", distributed_rank=local_rank)

    if rank == 0:
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = f"{cfg.Detector.name}_backend-{idist.backend()}-{idist.get_world_size()}_{now}"
        output_path = Path(cfg.output_path) / folder_name
        if not output_path.exists():
            output_path.mkdir(parents=True)
        cfg.output_path = output_path.as_posix()
        logger.info(f"Output path: {cfg.output_path}")

        if "cuda" in device.type:
            cfg["cuda_device_name"] = torch.cuda.get_device_name(local_rank)

    train_loader, test_loader = get_dataflow(cfg)

    cfg["num_iters_per_epoch"] = len(train_loader)
    cfg.checkpoint_every *= cfg["num_iters_per_epoch"]
    log_basic_info(logger, cfg)

    net, optimizer, criterion, lr_scheduler, priors = net_init(cfg)

    # Create trainer for current task
    trainer = create_trainer(net, optimizer, criterion, priors, lr_scheduler, train_loader.sampler, cfg, logger)

    # Let's now setup evaluator engine to perform model's validation and compute metrics

    if rank == 0:
        # Setup TensorBoard logging on trainer and evaluators. Logged values are:
        #  - Training metrics, e.g. running average loss values
        #  - Learning rate
        #  - Evaluation train/test metrics
        # evaluators = {"training": train_evaluator, "test": evaluator}
        tb_logger = common.setup_tb_logging(output_path, trainer, optimizer, # evaluators=evaluators, 
                                            log_every_iters=10)

    # Store 2 best models by validation accuracy starting from num_epochs / 2:
    # best_model_handler = Checkpoint(
    #     {"model": net},
    #     DiskSaver(cfg.output_path, require_empty=False),
    #     filename_prefix="best",
    #     n_saved=2,
    #     global_step_transform=global_step_from_engine(trainer),
    #     score_name="test_accuracy",
    #     score_function=Checkpoint.get_default_score_fn("Accuracy"),
    # )
    # evaluator.add_event_handler(
    #     Events.COMPLETED(lambda *_: trainer.state.epoch > config["num_epochs"] // 2), best_model_handler
    # )

    try:
        trainer.run(train_loader, max_epochs=cfg.max_epoch)
    except Exception as e:
        logger.exception("")
        raise e

    if rank == 0:
        tb_logger.close()
    

def net_init(cfg):
    net = build_detor(cfg.Detector)
    # Adapt model for distributed settings if configured
    net = idist.auto_model(net)

    optimizer = optim.SGD(net.parameters(), 
                          lr=cfg.lr, 
                          momentum=cfg.momentum, 
                          weight_decay=cfg.weight_decay,
                          nesterov=False)
    optimizer = idist.auto_optim(optimizer)

    criterion = MultiBoxLoss(cfg.num_classes, 0.35, True, 0, True, 7, 0.35, False)
    criterion = criterion.to(idist.device())
    with torch.no_grad():
        priors = PriorBox(cfg.min_sizes, cfg.steps, cfg.clip, image_size=cfg.Dataset.image_size).forward()
        priors = priors.to(idist.device())

    milestones = [cfg.num_iters_per_epoch * e for e in cfg.milestones]
    step_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    lr_scheduler = LRScheduler(step_scheduler)

    return net, optimizer, criterion, lr_scheduler, priors


def create_trainer(model, optimizer, criterion, priors, lr_scheduler, train_sampler, cfg, logger):

    device = idist.device()

    # Setup Ignite trainer:
    # - let's define training step
    # - add other common handlers:
    #    - TerminateOnNan,
    #    - handler to setup learning rate scheduling,
    #    - ModelCheckpoint
    #    - RunningAverage` on `train_step` output
    #    - Two progress bars on epochs and optionally on iterations

    with_amp = cfg.with_amp
    scaler = GradScaler(enabled=with_amp)

    def train_step(engine, batch):

        images, targets = batch[0], batch[1]
        if images.device != device:
            images = images.to(device, non_blocking=True)
            targets = [anno.to(device, non_blocking=True) for anno in targets]

        model.train()

        with autocast(enabled=with_amp):
            out = model(images)
            loss_l, loss_c, loss_lmk = criterion(out, priors, targets)
            loss = cfg.loc_lambda * loss_l + loss_c + loss_lmk

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if engine.state.iteration % 10 == 0:
            log_str = 'iter: {}, loc: {:.4f}, cls: {:.4f}, lmk: {:.4f}'.format(
                engine.state.iteration, loss_l.item(), loss_c.item(), loss_lmk.item())
            logger.info(log_str)
        return {
            "batch loss": loss.item(),
            "loc loss": loss_l.item(),
            "cls loss": loss_c.item(),
            "lmk loss": loss_lmk.item()
        }

    trainer = Engine(train_step)
    trainer.logger = logger

    to_save = {"trainer": trainer, "model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler}
    metric_names = ["batch loss", "loc loss", "cls loss", "lmk loss"]

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_sampler,
        to_save=to_save,
        save_every_iters=cfg.checkpoint_every,
        save_handler=DiskSaver(cfg.output_path, require_empty=False),
        lr_scheduler=lr_scheduler,
        output_names=metric_names if cfg.log_every_iters > 0 else None,
        with_pbars=False,
        clear_cuda_cache=False,
    )

    if cfg.resume_from is not None:
        checkpoint_fp = Path(cfg.resume_from)
        assert checkpoint_fp.exists(), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
        logger.info(f"Resume from a checkpoint: {checkpoint_fp.as_posix()}")
        checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    return trainer


def get_dataflow(cfg):
    # - Get train/test datasets
    if idist.get_rank() > 0:
        # Ensure that only rank 0 download the dataset
        idist.barrier()

    train_dataset = WiderFace(cfg.Dataset.label_file, preproc(cfg.Dataset.image_size[0], cfg.Dataset.image_mean))

    # Setup data loader also adapted to distributed config: nccl, gloo, xla-tpu
    train_loader = idist.auto_dataloader(train_dataset, 
                                         batch_size=cfg.batch_size, 
                                         num_workers=cfg.num_workers, 
                                         shuffle=True, 
                                         drop_last=True,
                                         collate_fn=detection_collate)

    # test_loader = idist.auto_dataloader(
    #     test_dataset, batch_size=2 * config["batch_size"], num_workers=config["num_workers"], shuffle=False,
    # )
    test_loader = None
    return train_loader, test_loader


def log_basic_info(logger, config):
    logger.info(f"Train {config.Detector.name} on {config.Dataset.name}")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        # explicitly import cudnn as
        # torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}")
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")

    logger.info("\n")
    logger.info(f"Configuration: \n{config}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")


if __name__ == '__main__':
    fire.Fire({"run": run})