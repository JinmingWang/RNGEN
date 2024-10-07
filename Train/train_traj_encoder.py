from Train.configs_traj_encoder import *


def train():
    # Dataset & DataLoader
    dataset = LaDeCachedDataset(DATA_DIR, max_trajs=N_TRAJS, set_name="train")
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True, collate_fn=LaDeCachedDataset.collate_fn,
                            drop_last=True)

    # Models
    encoder = Encoder(traj_len=TRAJ_LEN, encode_c=DIM_TRAJ_ENC, N_trajs=N_TRAJS).to(DEVICE)
    decoder = Decoder(N_trajs=N_TRAJS, traj_len=TRAJ_LEN, encode_c=DIM_TRAJ_ENC, decode_len=PATH_LEN).to(DEVICE)
    if CHECKPOINT is not None:
        encoder, decoder = loadModels(CHECKPOINT, encoder, decoder)
    loss_func = torch.nn.MSELoss()

    # Optimizer & Scheduler
    optimizer = AdamW([{"params": encoder.parameters(), "lr": LR},
                       {"params": decoder.parameters(), "lr": LR}], lr=LR)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE,
                                     min_lr=LR_REDUCE_MIN)

    # Prepare Logging
    os.makedirs(LOG_DIR)
    writer = SummaryWriter(log_dir=LOG_DIR)
    # plot_manager = PlotManager(5, 1, 1)
    mov_avg_loss = MovingAvg(MOV_AVG_LEN)
    global_step = 0
    best_loss = float("inf")
    plot_manager = PlotManager(5, 2, 2)

    for e in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Training Epoch {e + 1}/{EPOCHS}")
        for batch_trajs, batch_paths, batch_graph, batch_heatmap in pbar:
            optimizer.zero_grad()
            traj_enc = encoder(batch_trajs)
            recon_paths = decoder(traj_enc)

            loss = loss_func(recon_paths, batch_paths)

            loss.backward()
            optimizer.step()

            total_loss += loss
            global_step += 1
            mov_avg_loss.update(loss)

            pbar.set_postfix_str(f"loss={mov_avg_loss.get():.7f}, lr={optimizer.param_groups[0]['lr']:.5e}")

            if global_step % 5 == 0:
                writer.add_scalar("loss", mov_avg_loss.get(), global_step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)


        plot_manager.plotSegments(batch_graph[0], 0, 0, "Graph")
        plot_manager.plotTrajs(batch_trajs[0], 0, 1, "Trajs")
        plot_manager.plotTrajs(batch_paths[0], 1, 0, "Paths")
        plot_manager.plotTrajs(recon_paths[0], 1, 1, "Reconstructed Paths")
        writer.add_figure("Figure", plot_manager.getFigure(), global_step)

        saveModels(LOG_DIR + "last.pth", encoder, decoder)
        if total_loss < best_loss:
            best_loss = total_loss
            saveModels(LOG_DIR + "best.pth", encoder, decoder)

        lr_scheduler.step(total_loss)


if __name__ == "__main__":
    train()
