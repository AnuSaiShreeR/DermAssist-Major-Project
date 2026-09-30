# E4_resnet18_layer4_unfrozen_weighted_loss

## Purpose

Experiment E4 fine-tunes the last residual stage (`layer4`) of an ImageNet-pretrained ResNet18 while training a new HAM10000 classifier head. It uses class-weighted CrossEntropyLoss but does not use a balanced sampler.

## Configuration

- Model: ResNet18
- Pretrained weights: ImageNet / TorchVision `ResNet18_Weights.DEFAULT`
- Dataset: HAM10000 prepared as ImageFolder
- Classes: akiec, bcc, bkl, df, mel, nv, vasc
- Backbone policy: frozen except `layer4`
- Trainable layers: `layer4` and `fc`
- Loss: class-weighted CrossEntropyLoss
- Balanced sampler: no
- Batch size: 32
- Epochs: 15
- layer4 LR: 1e-05
- fc LR: 0.0001
- Weight decay: 0.0001
- Trainable parameters: 8,397,319
- Total parameters: 11,180,103

## Training class distribution

|    | class   |   count |
|---:|:--------|--------:|
|  0 | akiec   |     229 |
|  1 | bcc     |     360 |
|  2 | bkl     |     769 |
|  3 | df      |      81 |
|  4 | mel     |     779 |
|  5 | nv      |    4693 |
|  6 | vasc    |      99 |

## Class weights

|    | class   |    weight |
|---:|:--------|----------:|
|  0 | akiec   |  4.37305  |
|  1 | bcc     |  2.78175  |
|  2 | bkl     |  1.30225  |
|  3 | df      | 12.3633   |
|  4 | mel     |  1.28553  |
|  5 | nv      |  0.213388 |
|  6 | vasc    | 10.1154   |

## Test metrics

|    | metric            |    value |
|---:|:------------------|---------:|
|  0 | test_loss         | 0.712913 |
|  1 | accuracy          | 0.705256 |
|  2 | balanced_accuracy | 0.682298 |
|  3 | macro_f1          | 0.610715 |
|  4 | weighted_f1       | 0.733451 |

## Classification report

|              |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| akiec        |      0.6364 |   0.5714 |     0.6022 |   49      |
| bcc          |      0.4957 |   0.7403 |     0.5938 |   77      |
| bkl          |      0.6259 |   0.5576 |     0.5897 |  165      |
| df           |      0.4444 |   0.7059 |     0.5455 |   17      |
| mel          |      0.3178 |   0.6946 |     0.4361 |  167      |
| nv           |      0.9498 |   0.7336 |     0.8278 | 1006      |
| vasc         |      0.6071 |   0.7727 |     0.68   |   22      |
| accuracy     |      0.7053 |   0.7053 |     0.7053 |    0.7053 |
| macro avg    |      0.5824 |   0.6823 |     0.6107 | 1503      |
| weighted avg |      0.7998 |   0.7053 |     0.7335 | 1503      |

## Confusion matrix - counts

|            |   pred_akiec |   pred_bcc |   pred_bkl |   pred_df |   pred_mel |   pred_nv |   pred_vasc |
|:-----------|-------------:|-----------:|-----------:|----------:|-----------:|----------:|------------:|
| true_akiec |           28 |         13 |          2 |         2 |          4 |         0 |           0 |
| true_bcc   |            0 |         57 |          5 |         3 |          6 |         5 |           1 |
| true_bkl   |            6 |         16 |         92 |         1 |         33 |        16 |           1 |
| true_df    |            1 |          3 |          0 |        12 |          0 |         1 |           0 |
| true_mel   |            8 |          2 |         18 |         4 |        116 |        16 |           3 |
| true_nv    |            1 |         22 |         30 |         5 |        204 |       738 |           6 |
| true_vasc  |            0 |          2 |          0 |         0 |          2 |         1 |          17 |

## Confusion matrix - row normalized

|            |   pred_akiec |   pred_bcc |   pred_bkl |   pred_df |   pred_mel |   pred_nv |   pred_vasc |
|:-----------|-------------:|-----------:|-----------:|----------:|-----------:|----------:|------------:|
| true_akiec |        0.571 |      0.265 |      0.041 |     0.041 |      0.082 |     0     |       0     |
| true_bcc   |        0     |      0.74  |      0.065 |     0.039 |      0.078 |     0.065 |       0.013 |
| true_bkl   |        0.036 |      0.097 |      0.558 |     0.006 |      0.2   |     0.097 |       0.006 |
| true_df    |        0.059 |      0.176 |      0     |     0.706 |      0     |     0.059 |       0     |
| true_mel   |        0.048 |      0.012 |      0.108 |     0.024 |      0.695 |     0.096 |       0.018 |
| true_nv    |        0.001 |      0.022 |      0.03  |     0.005 |      0.203 |     0.734 |       0.006 |
| true_vasc  |        0     |      0.091 |      0     |     0     |      0.091 |     0.045 |       0.773 |

## Interpretation checklist

- Compare E4 against E1/E2/E3.
- Do not judge only by accuracy.
- Focus on macro F1, balanced accuracy, mel recall, akiec recall, bcc recall, and df/vasc behavior.
- Check whether unfreezing layer4 improved minority-class recognition.
- Check whether nv recall dropped too much.
- Review confident wrong predictions separately.
