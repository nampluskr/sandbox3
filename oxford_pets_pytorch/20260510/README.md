```bash
quad_regression/
│
├── configs/
│   ├── paths.yaml
│   ├── pretrain.yaml
│   └── finetune.yaml
│
├── src/
│   ├── config.py
│   ├── engine.py
│   ├── models/
│   │   ├── model_builder.py
│   │   └── quad_regressor.py
│   ├── datasets/                    # 미확정
│   └── utils/                       # 미확정
│
├── scripts/
│   ├── run_pretrain.py
│   └── run_finetune.py
│
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_engine.py
│   ├── models/
│   │   ├── test_model_builder.py
│   │   └── test_quad_regressor.py
│   ├── datasets/                    # 미확정
│   └── utils/                       # 미확정
│
├── notebooks/                       # .ipynb 파일만
│
├── experiments/                     # 탐색 / 디버깅 / 분석 / 프로토타이핑 .py
│   ├── explore/
│   ├── debug/
│   ├── prototype/
│   └── analysis/
│
├── outputs/
│   ├── pretrain/
│   └── finetune/
│       └── archive/
│
└── .gitignore
```
