# EPE-DARTS
EPE-NAS with DARTS+ for faster Neural Architecture Search


## Run example

Adjust the batch size if out of memory (OOM) occurs. It dependes on your gpu memory size and genotype.


- EPE-NAS search
```shell
python epe_search.py --dataset cifar100 - random_search
```

- Search
```shell
python search.py --name cifar10 --dataset cifar10
```

- Augment
```shell
# genotype from search results
python augment.py --name cifar10 --dataset cifar10 --genotype "Genotype(
    normal=[[('sep_conv_3x3', 0), ('dil_conv_5x5', 1)], [('skip_connect', 0), ('dil_conv_3x3', 2)], [('sep_conv_3x3', 1), ('skip_connect', 0)], [('sep_conv_3x3', 1), ('skip_connect', 0)]],
    normal_concat=range(2, 6),
    reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('max_pool_3x3', 0), ('skip_connect', 2)], [('skip_connect', 3), ('max_pool_3x3', 0)], [('skip_connect', 2), ('max_pool_3x3', 0)]],
    reduce_concat=range(2, 6)
)"
```


## Multi-GPU

This project supports multi-gpu. The larger batch size and learning rate are required to take advantage of multi-gpu.

- Search
```shell
python search.py --name cifar10-mg --dataset cifar10 --gpus 0,1,2,3 \
    --batch_size 256 --workers 16 --print_freq 10 \
    --w_lr 0.1 --w_lr_min 0.004 --alpha_lr 0.0012
```

- Augmentd
```shell
python augment.py --name cifar10-mg --dataset cifar10 --gpus 0,1,2,3 \
    --batch_size 384 --workers 16 --print_freq 50 --lr 0.1 \
    --genotype "Genotype(
    normal=[[('sep_conv_3x3', 0), ('dil_conv_5x5', 1)], [('skip_connect', 0), ('dil_conv_3x3', 2)], [('sep_conv_3x3', 1), ('skip_connect', 0)], [('sep_conv_3x3', 1), ('skip_connect', 0)]],
    normal_concat=range(2, 6),
    reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('max_pool_3x3', 0), ('skip_connect', 2)], [('skip_connect', 3), ('max_pool_3x3', 0)], [('skip_connect', 2), ('max_pool_3x3', 0)]],
    reduce_concat=range(2, 6)
)"
```

Simply, `--gpus all` makes to use all gpus.

### Cautions

It is well-known problem that the larger batch size causes lower generalization.
Note that although the [linear scaling rule](https://arxiv.org/pdf/1706.02677) prevents this problem somewhat, the generalization still could be bad.

Furthermore, we do not know about the scalability of DARTS, where larger batch size could be more harmful.
So, please pay attention to the hyperparameters when using multi-gpu.

## Results

The following results were obtained using the default arguments, except for the epochs. `--epochs 300` was used in MNIST and Fashion-MNIST.

| Dataset | Final validation acc | Best validation acc |
| ------- | -------------------- | ------------------- |
| MNIST         | 99.75% | 99.81% |
| Fashion-MNIST | 99.27% | 99.39% |
| CIFAR-10       | 97.17% | 97.23% |

97.17%, final validation accuracy in CIFAR-10, is the same number as the paper.

### Found architectures

```python
# CIFAR10
Genotype(
    normal=[[('sep_conv_3x3', 0), ('dil_conv_5x5', 1)], [('skip_connect', 0), ('dil_conv_3x3', 2)], [('sep_conv_3x3', 1), ('skip_connect', 0)], [('sep_conv_3x3', 1), ('skip_connect', 0)]],
    normal_concat=range(2, 6),
    reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('max_pool_3x3', 0), ('skip_connect', 2)], [('skip_connect', 3), ('max_pool_3x3', 0)], [('skip_connect', 2), ('max_pool_3x3', 0)]],
    reduce_concat=range(2, 6)
)

# FashionMNIST
Genotype(
    normal=[[('max_pool_3x3', 0), ('dil_conv_5x5', 1)], [('max_pool_3x3', 0), ('sep_conv_3x3', 1)], [('sep_conv_5x5', 1), ('sep_conv_3x3', 3)], [('sep_conv_5x5', 4), ('dil_conv_5x5', 3)]],
    normal_concat=range(2, 6),
    reduce=[[('sep_conv_3x3', 1), ('avg_pool_3x3', 0)], [('avg_pool_3x3', 0), ('skip_connect', 2)], [('skip_connect', 3), ('avg_pool_3x3', 0)], [('sep_conv_3x3', 2), ('skip_connect', 3)]],
    reduce_concat=range(2, 6)
)

# MNIST
Genotype(
    normal=[[('sep_conv_3x3', 0), ('dil_conv_5x5', 1)], [('sep_conv_3x3', 2), ('sep_conv_3x3', 1)], [('dil_conv_5x5', 3), ('sep_conv_3x3', 1)], [('sep_conv_5x5', 4), ('dil_conv_5x5', 3)]],
    normal_concat=range(2, 6),
    reduce=[[('dil_conv_3x3', 0), ('sep_conv_3x3', 1)], [('avg_pool_3x3', 0), ('skip_connect', 2)], [('dil_conv_5x5', 3), ('avg_pool_3x3', 0)], [('dil_conv_3x3', 1), ('max_pool_3x3', 0)]],
    reduce_concat=range(2, 6)
)
```

## Reference

* https://github.com/quark0/darts (official DARTS implementation)
* https://github.com/khanrc/pt.darts (pt.darts implementation)

