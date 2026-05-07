# Fine Tuning

```python
#####################################################################
# Configuration
#####################################################################
backbone = "resnet18"
image_size = 256

trained_dir = os.path.join(ROOT_DIR, "outputs", "training", backbone)
trained_weights_path = os.path.join(trained_dir, f"trained-{backbone}-img{image_size}.pth")
trained_config_path = os.path.join(trained_dir, f"trained-{backbone}-img{image_size}.yaml")

config = load_config(trained_config_path)
config.update({
    dataset=["data1", "data2"],
    seed=42,
    batch_size=16,
    max_epoch=30,
    early_stop=False,
    no_logging=False,
})

tuning_dir=os.path.join(ROOT_DIR, "outputs", "tuning", backbone)
experiment = f"tunned-{backbone}-img{image_size}"
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
# Fine tuning (output_dir: log / weights / config)
#####################################################################
model = build_model(
    backbone=config["backbone"],
    output_dim=8,
    pretrained=False,
)
load_weights(model_trained, trained_weights_path)

for learning_rate in [1e-4, 1e-5, 1e-6]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    trainer = Regressor(model, optimizer=optimizer)
    fit(trainer, max_epoch=10, logger=logger)

save_weights(model, tunned_weights_path)
```
