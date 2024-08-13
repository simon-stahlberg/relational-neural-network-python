# Relational Neural Network for Classical Planning

This repository implements the relational neural network architecture presented in *Ståhlberg, Bonet, Geffner, ICAPS 2022* (https://ojs.aaai.org/index.php/ICAPS/article/view/19851/19610).
The implementation is not strictly faithful to the original and includes a few optimizations to reduce training time, such as residual connections.
The loss function also includes an auxiliary term to help reduce the number of training steps.

The purpose of this repository is to serve as a starting point for developing new approaches and is thus very barebones.
It contains a very simple training loop without any optimizations like learning rate schedulers, curriculum learning, gradient clipping, and so on.
The testing code is also very simple, greedily following the successor state with the lowest predicted value.
Cycle detection is commonly used, but it is not implemented here to keep the code simple.

## Dependencies

The two main dependencies are `torch` and `pymimir`, both of which are available via pip.
Please see https://pytorch.org/ for instructions on how to install `torch`.
The `pymimir` package can be installed by running `pip install pymimir`.

## Example

A pre-trained model for Blocks is included in the repository to demonstrate the architecture's capabilities.
This model can be found in `example/blocks.pth`.
This file contains the hyperparameters to initialize the model, the model weights, and the state of the optimizer.
The last part is useful for pausing and resuming training.
Training and testing files for Blocks are located in `example/blocks` and `example/blocks/test`, respectively.

### Training

You can call `train.py` with `--help` to see the possible arguments for the training procedure.
The only required input is a path to either a single PDDL problem file or a directory containing them.

```
$ python3 train.py --help
usage: train.py [-h] --input INPUT [--model MODEL] [--embedding_size EMBEDDING_SIZE] [--layers LAYERS] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE] [--num_epochs NUM_EPOCHS]

Settings for training

options:
  -h, --help            show this help message and exit
  --input INPUT         Path to the training dataset
  --model MODEL         Path to a pre-trained model to continue training from
  --embedding_size EMBEDDING_SIZE
                        Dimension of the embedding vector for each object
  --layers LAYERS       Number of layers in the model
  --batch_size BATCH_SIZE
                        Number of samples per batch
  --learning_rate LEARNING_RATE
                        Learning rate for the training process
  --num_epochs NUM_EPOCHS
                        Number of epochs for the training process
```

The following command was used to train the model for Blocks.

```
$ python3 train.py --input example/blocks
Torch: 2.3.0+cu121
GPU is available. Using GPU: NVIDIA GeForce RTX 3090
Parsing files...
Generating state spaces...
> Expanding: example/blocks/probBLOCKS-4-0.pddl
- # States: 125
> Expanding: example/blocks/probBLOCKS-4-1.pddl
- # States: 125
...
> Expanding: example/blocks/probBLOCKS-8-2.pddl
- # States: 695417
> Expanding: example/blocks/probBLOCKS-9-0.pddl
- Skipped
> Expanding: example/blocks/probBLOCKS-9-1.pddl
- Skipped
> Expanding: example/blocks/probBLOCKS-9-2.pddl
- Skipped
Creating state samplers...
Creating a new model and optimizer...
Creating datasets...
Training model...
[1/1000; 100/10000] Loss: 2.1691
[1/1000; 200/10000] Loss: 1.7916
[1/1000; 300/10000] Loss: 1.2292
...
```

During the training process, the network is evaluated against the validation set after each epoch.
Two models are saved: `latest.pth` and `best.pth`.
The first model, `latest.pth`, is saved after every epoch.
The second model, `best.pth`, is the one with the lowest error on the validation set.

### Testing

The pre-trained model can be run on an instance, even if it is not part of the training set, using the following command:

```
$ python3 plan.py --model example/blocks.pth --input example/blocks/test/probBLOCKS-17-0.pddl
Torch: 2.3.0+cu121
GPU is available. Using GPU: NVIDIA GeForce RTX 3090
Creating parser...
Loading model... (example/blocks.pth)
49.440: (unstack l f)
48.403: (put-down l)
47.138: (unstack g d)
...
1.635: (stack n l)
0.929: (pick-up q)
-0.018: (stack q n)
Found a solution of length 46!
1: (unstack l f)
2: (put-down l)
3: (unstack g d)
...
44: (stack n l)
45: (pick-up q)
46: (stack q n)
```

In the output, the selected action is printed along with the predicted value of the resulting successor state.
Here, the network predicted the final solution would require at least 49 steps, but it ended up taking 46 steps.
If a plan is found, it is also printed at the end.

It is also possible to use the learned model as a heuristic function for A*:

```
$ python3 search.py --model example/blocks.pth --input example/blocks/test/probBLOCKS-17-0.pddl
Torch: 2.3.0+cu121
GPU is available. Using GPU: NVIDIA GeForce RTX 3090
Creating parser...
Loading model... (example/blocks.pth)
[f = 50.755] Expanded: 0; Generated: 0
[Final] Expanded: 50; Generated: 338
Solved using 46 actions
1: (unstack l f)
2: (put-down l)
3: (unstack g d)
...
44: (stack n l)
45: (pick-up q)
46: (stack q n)
```

However, there is no way to batch, so the GPU may be underutilized.
