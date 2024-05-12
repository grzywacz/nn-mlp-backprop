# Neural Networks

Implementation of MLP with backpropagation.

## Requirements

python3

```
python3 -m pip install -r requirements.txt
```

## Usage

Print usage help message:

```
python3 main.py --help
```

Run training on UCI Wine dataset:

```
python3 main.py
```

Run training on UCI Breast Cancer dataset:

```
python3 main.py --dataset breast_cancer --hidden_layers 32 16 --activation tanh --batch_size 64 --epochs 25 --learning_rate 0.01
```
