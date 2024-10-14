# RNGEN
Just for private project

## Good Ideas
| Idea                                                    | Description                                |
|---------------------------------------------------------|--------------------------------------------|
| Trajs to sub-trajs encoder                              | Relation among sub-trajs is more clear     |
| Hungarian loss                                          | Items in graph are unordered               |
| Compute loss against $x_{t-1}$ instead of $\varepsilon$ | So we can use Hungarian loss               |
| Icrease # of attn layers in encoder                     | To better capture relation among sub-trajs |
| Segments-gen diffusion model                            | So we generate only segments               |
| Use of torch.compile                                    | Saves training time                        |
| Also predict padding mask dimension                     | So model can also predict # of items       |

## Bad Ideas
| Idea                                                    | Description                                |
|---------------------------------------------------------|--------------------------------------------|
| Increase diffusion T from 500 to 1000                   | The performance is decreased               |
| Increase dims in attn in encoder                        | No visible improvement                     |
| Set # of trajs to 64                                    | Costly in memory, very slight improvement  |

## TODO Ideas
- Change number of trajs from 64 to 32. What does more trajs mean? More burden for the model or more information? Now we feed 64 trajs to the model, the memory cost is very high. In contrast, with 32 trajs the memory cost drops significantly. What is the effect on the generation results?