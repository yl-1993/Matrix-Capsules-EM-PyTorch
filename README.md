# Matrix Capsules with EM Routing
A PyTorch implementation of [Matrix Capsules with EM Routing](https://openreview.net/pdf?id=HJWLfGWRb)

## Usage
1. Install [PyTorch](http://pytorch.org/)

2. Start training (default: MNIST)
```bash
python train.py
```

Note that master is upgraded to be compatiable with PyTorch `0.4.0`.
If you want to use the old version of PyTorch, please
```bash
git checkout 0.3.1.post3
```

## MNIST experiments

The experiments are conducted on TitanXP.
Specific setting is `lr=0.01`, `batch_size=128`, `weight_decay=0`, Adam optimizer, without data augmentation.
The paper does not mention the specific scheduler for `inverse_temperature`, it is fixed to `0.001` in our setting.
As our experiments shown, `\lambda` between `1e-2` and `1e-4` achieves similar results. A large lambda may prevent the model from convergence.

Following is the result after 30 epochs training:

| Arch | Iters | Coord Add | Loss | BN | Test Accuracy |
| ---- |:-----:|:---------:|:----:|:--:|:-------------:|
| A=64, B=8, C=D=16 | 1 | Y | Spread    | Y | 97.1 |
| A=64, B=8, C=D=16 | 2 | Y | Spread    | Y | 99.1 |
| A=64, B=8, C=D=16 | 3 | Y | Spread    | Y | 97.5 |
| A=64, B=8, C=D=16 | 2 | N | Spread    | Y | 99.0 |
| A=64, B=8, C=D=16 | 2 | Y | Spread    | N | 98.9 |
| A=64, B=8, C=D=16 | 2 | Y | Cross-Ent | Y | 97.8 |
| A=B=C=D=32        | 2 | Y | Spread    | Y | 99.3 |

The training time of `A=64, B=8, C=D=16` for a 128 batch is around `1.05s`.
The training time of `A=B=C=D=32` for a 32 batch is around `1.45s`.

## smallNORB experiments

```bash
python train.py --dataset smallNORB --batch-size 32 --test-batch-size 256
```

As the paper suggests, the image is resized to 48x48, followed by randomly cropping a 32x32 patch and randomly changing brightness and contrast.
Since BN is used after first normal conv layer, we do not normalize the input.

Following is the result after 50 epochs training:

| Arch | Iters | Coord Add | Loss | BN | Test Accuracy |
| ---- |:-----:|:---------:|:----:|:--:|:-------------:|
| A=64, B=8, C=D=16 | 1 | Y | Spread    | Y | 74.81 |
| A=64, B=8, C=D=16 | 2 | Y | Spread    | Y | 89.52 |
| A=64, B=8, C=D=16 | 3 | Y | Spread    | Y | 82.55 |
| A=B=C=D=32        | 2 | Y | Spread    | Y | 90.03 |

A weird thing is that large batch size seems to result in poor result.

## Reference
- https://github.com/shzygmyx/Matrix-Capsules-pytorch
- https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow
- https://github.com/gyang274/capsulesEM
