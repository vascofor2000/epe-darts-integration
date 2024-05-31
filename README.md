# DADARTS
Discretization Aware Differentiable ARchiTecture Search

# Abstract
Neural Architecture Search has garnered significant attention for democratizing access to Deep neural networks (DNNs) by reducing the costs associated with developing new neural network architectures. These costs are minimized through algorithms that efficiently identify optimal architectures for specific problems, mitigating the need for extensive trial-and-error and expert knowledge. Despite this, the training duration of these algorithms can still be lengthy depending on the techniques employed. Differentiable ARchiTecture Search (DARTS) emerged as an efficient and elegant solution, offering an end-to-end differentiable approach that can be trained like a conventional DNN. However, DARTS has encountered robustness issues, which stem from the usage of heuristics that appear counter-intuitive to the algorithm's overall design. In this project, I present three techniques that eliminate the need for this heuristic, enhancing the algorithm's intuitiveness and robustness for application across various domains.

# Run the architecture search
- Discretization at Validation Time
```shell
python search.py -name "some_name" -dataset cifar10 -hd_on_weights -workers 4
```

- Differentiable TopK on alphas
```shell
python search.py -name "some_name" -dataset cifar10 -topk_on_alphas -linear_temperature_increase -linear_temperature_max 200 -workers 4
```

- Differentiable TopK on alphas and weights
```shell
python search.py -name "some_name" -dataset cifar10 -topk_on_alphas -topk_on_weights -linear_temperature_increase -linear_temperature_max 200 -workers 4
```

- Discretization Additional Loss
```shell
python search.py -name "some_name" -dataset cifar10 -dal_on_alphas -workers 4
```

- DADARTS
```shell
python search.py -name "some_name" -dataset cifar10 -dal_on_alphas -topk_on_alphas -linear_temperature_increase -linear_temperature_max 200 -hd_on_weights -workers 4
```

# Run the training of the obtained architecture
```shell
python augment.py -name "some_name" -dataset cifar10 -workers 4 -genotype "Genotype(normal=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)], [('max_pool_3x3', 2), ('max_pool_3x3', 0)], [('max_pool_3x3', 3), ('max_pool_3x3', 0)], [('max_pool_3x3', 4), ('max_pool_3x3', 0)]], normal_concat=[2, 3, 4, 5], reduce=[[('dil_conv_3x3', 0), ('dil_conv_3x3', 1)], [('max_pool_3x3', 2), ('sep_conv_3x3', 0)], [('max_pool_3x3', 2), ('max_pool_3x3', 3)], [('max_pool_3x3', 2), ('max_pool_3x3', 3)]], reduce_concat=[2, 3, 4, 5])"
```