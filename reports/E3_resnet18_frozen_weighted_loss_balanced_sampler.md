# E3_resnet18_frozen_weighted_loss_balanced_sampler
## Configuration
- Model: ImageNet-pretrained ResNet18
- Backbone frozen: True
- Unfreeze layer4: False
- Class-weighted loss: True
- Weighted sampler: True
- Epochs: 10
- Batch size: 32
- Learning rate: 0.001
- Elapsed minutes: 3.99

## Training class distribution

- akiec: 229
- bcc: 360
- bkl: 769
- df: 81
- mel: 779
- nv: 4693
- vasc: 99

## Class weights

- akiec: 4.3731
- bcc: 2.7817
- bkl: 1.3022
- df: 12.3633
- mel: 1.2855
- nv: 0.2134
- vasc: 10.1154

## Test metrics

- Accuracy: 0.3646
- Balanced accuracy: 0.5340
- Macro F1: 0.3098
- Weighted F1: 0.4251
- Test loss: 2.0248

## Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| akiec | 0.2143 | 0.4286 | 0.2857 | 49 |
| bcc | 0.3739 | 0.5584 | 0.4479 | 77 |
| bkl | 0.4000 | 0.4364 | 0.4174 | 165 |
| df | 0.0759 | 0.6471 | 0.1358 | 17 |
| mel | 0.2229 | 0.4551 | 0.2992 | 167 |
| nv | 0.9871 | 0.3032 | 0.4639 | 1006 |
| vasc | 0.0635 | 0.9091 | 0.1187 | 22 |

## Confusion matrix

|            |   pred_akiec |   pred_bcc |   pred_bkl |   pred_df |   pred_mel |   pred_nv |   pred_vasc |
|:-----------|-------------:|-----------:|-----------:|----------:|-----------:|----------:|------------:|
| true_akiec |           21 |         12 |          2 |        11 |          2 |         0 |           1 |
| true_bcc   |            8 |         43 |          3 |         8 |          4 |         0 |          11 |
| true_bkl   |           21 |         15 |         72 |        14 |         22 |         2 |          19 |
| true_df    |            0 |          4 |          0 |        11 |          1 |         0 |           1 |
| true_mel   |           23 |         11 |         28 |         9 |         76 |         2 |          18 |
| true_nv    |           25 |         28 |         75 |        92 |        236 |       305 |         245 |
| true_vasc  |            0 |          2 |          0 |         0 |          0 |         0 |          20 |

## Normalized confusion matrix, row-wise

|            |   pred_akiec |   pred_bcc |   pred_bkl |   pred_df |   pred_mel |   pred_nv |   pred_vasc |
|:-----------|-------------:|-----------:|-----------:|----------:|-----------:|----------:|------------:|
| true_akiec |        0.429 |      0.245 |      0.041 |     0.224 |      0.041 |     0     |       0.02  |
| true_bcc   |        0.104 |      0.558 |      0.039 |     0.104 |      0.052 |     0     |       0.143 |
| true_bkl   |        0.127 |      0.091 |      0.436 |     0.085 |      0.133 |     0.012 |       0.115 |
| true_df    |        0     |      0.235 |      0     |     0.647 |      0.059 |     0     |       0.059 |
| true_mel   |        0.138 |      0.066 |      0.168 |     0.054 |      0.455 |     0.012 |       0.108 |
| true_nv    |        0.025 |      0.028 |      0.075 |     0.091 |      0.235 |     0.303 |       0.244 |
| true_vasc  |        0     |      0.091 |      0     |     0     |      0     |     0     |       0.909 |

## Files written

- Raw confusion matrix: `reports/E3_resnet18_frozen_weighted_loss_balanced_sampler_confusion_matrix.csv`
- Normalized confusion matrix: `reports/E3_resnet18_frozen_weighted_loss_balanced_sampler_confusion_matrix_normalized.csv`
- Epoch history: `reports/E3_resnet18_frozen_weighted_loss_balanced_sampler_history.json`
