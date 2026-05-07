# Inference

```python
#####################################################################
# Configuration
#####################################################################
backbone = "resnet18"
image_size = 256

trained_dir = os.path.join(ROOT_DIR, "outputs", "tuning", backbone)
trained_weights_path = os.path.join(trained_dir, f"tuned-{backbone}-img{image_size}.pth")
trained_config_path = os.path.join(trained_dir, f"tuned-{backbone}-img{image_size}.yaml")

config = load_config(trained_config_path)
config["dataset"] = ["data1", "data2"]

#####################################################################
# Data Loading
#####################################################################
default = load_config(os.path.join(ROOT_DIR, "configs", "default.yaml"))
test_transform=get_transform(split="test", img_size=config["image_size"])

if isinstance(config["dataset"], srt):
    dataset = config["dataset"]
    test_dataset = get_dataset(
        image_dir=default[dataset]["test_image_dir"],
        csv_path=default[dataset]["test_csv_path"],
        transform=test_transform,
    )
elif isinstance(config["dataset"], list):
    test_dataset = merge_datasets([
        get_dataset(
            image_dir=default[dataset]["test_image_dir"],
            csv_path=default[dataset]["test_csv_path"],
            transform=test_transform,
        ) for dataset in config["dataset"]
    ])

test_loader = get_dataloader(
    dataset=test_dataset,
    split="test",
    batch_size=config["batch_size"],
)

#####################################################################
# Inference
#####################################################################
model = build_model(
    backbone=config["backbone"],
    output_dim=8,
    pretrained=False,
)
load_weights(model_trained, train_weights_path)

if isinstance(config["dataset"], srt):
    for sample in test_loader.dataset.samples:
        image_path = sample["image_path"]
        target = sample["bbox"]
        pred = predict(model, image_path, image_size=config["image_size"], transform=test_transform)
        show_image_poly(image_path, target, pred=pred)

elif isinstance(config["dataset"], list):
    all_samples = []
    for dataset in test_loader.dataset.datasets:
        all_samples.extend(dataset.samples)

    for sample in random.sample(all_samples, k=20):
        image_path = sample["image_path"]
        target = sample["bbox"]
        pred = predict(model, image_path, image_size=config["image_size"])
        show_image_poly(image_path, target, pred=pred)
```
