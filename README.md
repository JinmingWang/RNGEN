# RNGEN
Just for private project

## Good Ideas
| Idea                                                       | Description                                |
|------------------------------------------------------------|--------------------------------------------|
| Trajs to sub-trajs encoder                                 | Relation among sub-trajs is more clear     |
| Hungarian loss                                             | Items in graph are unordered               |
| Compute loss against $x_{t-1}$ instead of $\varepsilon$    | So we can use Hungarian loss               |
| Icrease # of attn layers in encoder                        | To better capture relation among sub-trajs |
| Segments-gen diffusion model                               | So we generate only segments               |
| Use of torch.compile                                       | Saves training time                        |
| Also predict padding mask dimension                        | So model can also predict # of items       |
| Predict $x_0$ instead of $\varepsilon$                     | Better performance                         |
| Predict $x_0$ and $\varepsilon$ together, match with $x_0$ | Better performance                         |
| Fuse t using addition, more processing to t                | Better performance                         |

## Bad Ideas
| Idea                                                    | Description                                |
|---------------------------------------------------------|--------------------------------------------|
| Increase diffusion T from 500 to 1000                   | The performance is decreased               |
| Increase dims in attn in encoder                        | No visible improvement                     |
| Set # of trajs to 64                                    | Costly in memory, very slight improvement  |
| use Conv + non-overlap-split in subtraj encoder         | Cause performance decrease                 |
| Split subtraj as 8 points                               | Cause performance decrease                 |

## TODO Ideas
- 测试究竟需要多少subtraj才是最好的

## 关于轨迹规模
| N_trajs | L_subtraj | D_subtraj | Result          | Cost |
|---------|-----------|-----------|-----------------|------|
| 64      | 16        | 32        | Good            | 4x   |
| 32      | 16        | 32        | Slight Decrease | 2x   |
| 32      | 8         | 16        | Decrease        | 4x   |
| 32      | 32        | 64        | Slight Decrease | 1x   |

目前来看，还是N_trajs=32，且有overlap的版本最经济。没有占用很大量的显存，效果也不错。
下一步可以尝试增加一些attention layer以及head数量，看看效果如何。
另外，也可以训练一个SegmentsVAE，它可以Encode Segments，并Decode出 Segments (或者直接Decode Nodes和adjacency matrix)。这样恢复的效果可能会更好。