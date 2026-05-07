# Training

```python
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_epoch", type=int, default=10)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--no_logging", action="store_true")
    return parser.parse_args()

def set_config(args):
    config = {
        backbone=args.backbone,         # resnet18
        seed=args.seed,                 # 42
        image_size=args.image_size,     # 256
        batch_size=args.batch_size,     # 16
        max_epoch=args.max_epoch,       # 100
        early_stop=args.early_stop,     # True
        no_logging=args.no_logging,     # False
    }    
    config["dataset"] = ["midv2020", "smartdoc"]
    return config
```

```python
#####################################################################
# Configuration
#####################################################################
args = parse_args()
conifg = set_config(args)
# config = {
#     backbone="resnet18",
#     seed=42,
#     image_size=256,
#     batch_size=16,
#     max_epoch=100,
#     early_stop=True,
#     no_logging=False,
# }
config["dataset"] = ["midv2020", "smartdoc"]

output_dir=os.path.join(ROOT_DIR, "outputs", "training", config["backbone"])
experiment = f"trained-{config['backbone']}-img{config['image_size']}"
log_path = os.path.join(output_dir, f"{experiment}.log")
weights_path = os.path.join(output_dir, f"{experiement}.pth")
config_path = os.path.join(output_dir, f"{experiment}.yaml")

save_config(config, config_path)
set_seed(seed=config["seed"])
logger = get_logger(log_path)

#####################################################################
# Data Loading
#####################################################################
logger = get_logger(log_path)
set_seed(seed=config["seed"])

default = load_config(os.path.join(ROOT_DIR, "configs", "default.yaml"))
if isinstance(config["dataset"], srt):
    dataset = config["dataset"]
    train_dataset = get_dataset(
        image_dir=default[dataset]["train_image_dir"],
        csv_path=default[dataset]["train_csv_path"],
        transform=get_transform(split="train", img_size=config["image_size"]),
    )
    test_dataset = get_dataset(
        image_dir=default[dataset]["test_image_dir"],
        csv_path=default[dataset]["test_csv_path"],
        transform=get_transform(split="test", img_size=config["image_size"]),
    )
elif isinstance(config["dataset"], list):
    train_dataset = merge_datasets([
        get_dataset(
            image_dir=default[dataset]["train_image_dir"],
            csv_path=default[dataset]["train_csv_path"],
            transform=get_transform(split="train", img_size=config["image_size"]),
        ) for dataset in config["dataset"]
    ])
    test_dataset = merge_datasets([
        get_dataset(
            image_dir=default[dataset]["test_image_dir"],
            csv_path=default[dataset]["test_csv_path"],
            transform=get_transform(split="test", img_size=config["image_size"]),
        ) for dataset in config["dataset"]
    ])

train_loader = get_dataloader(
    dataset=train_dataset,
    split="train",
    batch_size=config["batch_size"],
)

test_loader = get_dataloader(
    dataset=test_dataset,
    split="test",
    batch_size=config["batch_size"],
)

#####################################################################
# Training (output_dir: log / weights / config)
#####################################################################
model = build_model(
    backbone=config["backbone"],
    backbone_dir=default["backbone_dir"],
    output_dim=8,
    pretrained=True,
)
trainer = Regressor(model)
if config["early_stop"]:
    fit_early_stop(trainer, max_epoch=config["max_epoch"], valid_loder=test_loader, logger=logger)
else:
    fit(trainer, max_epoch=config["max_epochs"], valid_loader=test_loader, logger=logger)

save_weights(model, weights_path=config["weights_path"])

```
