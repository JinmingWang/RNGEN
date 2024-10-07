from Train.configs_node_egde_model import *


def train():
    # Dataset & DataLoader
    dataset = LaDeCachedDataset(DATA_DIR, max_trajs=N_TRAJS, set_name="train")
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True, collate_fn=LaDeCachedDataset.collate_fn, drop_last=True)

    # Models
    encoder = Encoder(traj_len=TRAJ_LEN, encode_c=DIM_TRAJ_ENC, N_trajs=N_TRAJS).to(DEVICE)
    diffusion_net = DiffusionNetwork(num_nodes=N_NODES, traj_encoding_c=DIM_TRAJ_ENC, traj_num=N_TRAJS, T=T).to(DEVICE)
    encoder, diffusion_net = loadModels("Runs/NodeEdgeModel_2024-10-07_04-50-30/last.pth", encoder, diffusion_net)
    ddpm = DDPM(BETA_MIN, BETA_MAX, T, DEVICE, "quadratic")
    # loss_func = torch.nn.MSELoss()
    loss_func = HungarianLoss('l1')

    # Optimizer & Scheduler
    optimizer = AdamW([{"params": diffusion_net.parameters(), "lr": LR_DIFFUSION},
                       {"params": encoder.parameters(), "lr": LR_ENCODER}], lr=LR_DIFFUSION)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE, min_lr=LR_REDUCE_MIN)

    # Prepare Logging
    os.makedirs(LOG_DIR)
    writer = SummaryWriter(log_dir=LOG_DIR)
    # plot_manager = PlotManager(5, 1, 1)
    mov_avg_loss = MovingAvg(MOV_AVG_LEN)
    global_step = 0
    best_loss = float("inf")

    for e in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Training Epoch {e + 1}/{EPOCHS}")
        for batch_traj, batch_path, batch_graph, batch_heatmap in pbar:

            # (B, N_nodes, 3), (B, N_nodes, N_nodes), (B,)
            batch_nodes, batch_adj_mat, batch_node_count = LaDeCachedDataset.SegmentsToNodesAdj(batch_graph, N_NODES)

            node_noise = torch.randn_like(batch_nodes)
            adj_mat_noise = torch.randn_like(batch_adj_mat)
            setPaddingToZero(batch_node_count, sequence=node_noise)
            setPaddingToZero(batch_node_count, adj_mat=adj_mat_noise)

            t = torch.randint(0, T, (B,)).to(DEVICE)
            noisy_batch_nodes = ddpm.diffusionForward(batch_nodes, t, node_noise)
            noisy_batch_adj_mat = ddpm.diffusionForward(batch_adj_mat, t, adj_mat_noise)

            s = t - 1
            less_noisy_batch_nodes = torch.zeros_like(noisy_batch_nodes)
            less_noisy_batch_nodes[t!=0] = ddpm.diffusionForward(batch_nodes[t!=0], s[t!=0], node_noise[t!=0])
            less_noisy_batch_nodes[t==0] = batch_nodes[t==0]

            less_noisy_batch_adj_mat = torch.zeros_like(noisy_batch_adj_mat)
            less_noisy_batch_adj_mat[t!=0] = ddpm.diffusionForward(batch_adj_mat[t!=0], s[t!=0], adj_mat_noise[t!=0])
            less_noisy_batch_adj_mat[t==0] = batch_adj_mat[t==0]

            optimizer.zero_grad()
            traj_enc = encoder(batch_traj)
            pred_node_noise, pred_adj_mat_noise = diffusion_net(noisy_batch_nodes, noisy_batch_adj_mat, traj_enc, t)
            setPaddingToZero(batch_node_count, sequence=pred_node_noise)
            setPaddingToZero(batch_node_count, adj_mat=pred_adj_mat_noise)

            pred_node = ddpm.diffusionBackwardStep(noisy_batch_nodes, t, pred_node_noise, disable_noise=True)
            pred_adj_mat = ddpm.diffusionBackwardStep(noisy_batch_adj_mat, t, pred_adj_mat_noise, disable_noise=True)

            loss = loss_func(pred_node, less_noisy_batch_nodes)#  + loss_func(pred_adj_mat, less_noisy_batch_adj_mat)

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2.0)
            torch.nn.utils.clip_grad_norm_(diffusion_net.parameters(), 2.0)

            optimizer.step()

            total_loss += loss.item()
            global_step += 1
            mov_avg_loss.update(loss)

            pbar.set_postfix_str(f"loss={mov_avg_loss.get():.5e}, lr={optimizer.param_groups[0]['lr']:.5e}")

            if global_step % LOG_INTERVAL == 0:
                writer.add_scalar("loss", mov_avg_loss.get(), global_step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)

            if global_step % EVAL_INTERVAL == 0:
                figure, eval_loss = eval(batch_nodes, batch_adj_mat, batch_traj, batch_node_count, encoder, diffusion_net, ddpm)
                writer.add_figure("Evaluation", figure, global_step)
                writer.add_scalar("eval_loss", eval_loss.item(), global_step)

        saveModels(LOG_DIR + "last.pth", encoder, diffusion_net)
        if total_loss < best_loss:
            best_loss = total_loss
            saveModels(LOG_DIR + "best.pth", encoder, diffusion_net)

        lr_scheduler.step(total_loss)


def eval(batch_nodes, batch_adj_mat, batch_traj, batch_node_count, encoder, diffusion_net, ddpm):
    encoder.eval()
    diffusion_net.eval()

    with torch.no_grad():
        traj_enc = encoder(batch_traj)
        node_noise = torch.randn_like(batch_nodes)
        adj_mat_noise = torch.randn_like(batch_adj_mat)
        setPaddingToZero(batch_node_count, sequence=node_noise)
        setPaddingToZero(batch_node_count, adj_mat=adj_mat_noise)

        def pred_func(noisy_contents: List[Tensor], t: Tensor) -> List[Tensor]:
            setPaddingToZero(batch_node_count, sequence=noisy_contents[0])
            setPaddingToZero(batch_node_count, adj_mat=noisy_contents[1])
            node_noise_pred, adj_mat_noise_pred = diffusion_net(*noisy_contents, traj_enc, t)
            setPaddingToZero(batch_node_count, sequence=node_noise_pred)
            setPaddingToZero(batch_node_count, adj_mat=adj_mat_noise_pred)
            return [node_noise_pred, adj_mat_noise_pred]

        nodes, adj_mat = ddpm.diffusionBackward([node_noise, adj_mat_noise], pred_func)
        adj_mat = (adj_mat > 0.5).to(torch.int32)

    node_loss = HungarianLoss()(nodes, batch_nodes)

    plot_manager = PlotManager(5, 2, 2)

    plot_manager.plotNodesWithAdjMat(batch_nodes[0], batch_adj_mat[0], 0, 0, "Original Graph")
    plot_manager.plotNodesWithAdjMat(nodes[0], torch.zeros_like(adj_mat[0]), 0, 1, "Reconstructed Graph")
    plot_manager.plotTrajs(batch_traj[0], 1, 0, "Trajectories")

    encoder.train()
    diffusion_net.train()

    return plot_manager.getFigure(), node_loss



if __name__ == "__main__":
    train()
