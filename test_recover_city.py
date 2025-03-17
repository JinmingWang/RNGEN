import matplotlib.pyplot as plt
from TrainEvalTest.Utils import *
from TrainEvalTest.Metrics import *
from Diffusion import DDIM
from tqdm import tqdm

import torch

from Dataset import DEVICE, RoadNetworkDataset
from Models import WGVAE_new, TWDiT, AD_Linked_Net, NodeExtractor, DFDRUNet, UNet2D, GraphusionVAE, Graphusion

titles = ["hungarian_mae", "hungarian_mse", "chamfer_mae", "chamfer_mse", "diff_seg_count", "diff_seg_len"]


def prepareDataset(city: str):

    dataset = RoadNetworkDataset(folder_path=f"Dataset/{city}",
                                     batch_size=1,
                                     drop_last=True,
                                     set_name="all",
                                     permute_seq=False,
                                     enable_aug=False,
                                     shuffle=False,
                                     img_H=256,
                                     img_W=256,
                                     need_heatmap=True,
                                     need_image=True,
                                     need_nodes=True
                                     )

    # --- Get range of all points --------------------------

    all_points = dataset.trajs * dataset.std_norm + dataset.mean_norm
    lng_min = all_points[..., 0].min()
    lng_max = all_points[..., 0].max()
    lat_min = all_points[..., 1].min()
    lat_max = all_points[..., 1].max()
    lng_range = lng_max - lng_min
    lat_range = lat_max - lat_min
    coverage_mask = torch.zeros(1000, 1000, dtype=torch.int32, device=DEVICE)

    adopted_sample_ids = []
    bboxes = []

    # --- Eliminate Duplicate Bboxes ----------------------------------

    for bi, batch in enumerate(tqdm(dataset)):
        batch_points = (batch["trajs"] * batch["std_point"] + batch["mean_point"]).view(-1, 2).cpu().numpy()
        left, bottom = batch_points.min(axis=0)
        right, top = batch_points.max(axis=0)
        width = right - left
        height = top - bottom

        left_id = int((left - lng_min) / lng_range * 1000)
        right_id = int((right - lng_min) / lng_range * 1000)
        bottom_id = int((bottom - lat_min) / lat_range * 1000)
        top_id = int((top - lat_min) / lat_range * 1000)

        bboxes.append((left_id, right_id, bottom_id, top_id))

        if not torch.all(coverage_mask[bottom_id:top_id, left_id:right_id] > 0):
            coverage_mask[bottom_id:top_id, left_id:right_id] += 1
            adopted_sample_ids.append(bi)

    print("First Round adopted sample ids:", len(adopted_sample_ids))

    redundant_bi = []
    for bi in adopted_sample_ids:
        l, r, b, t = bboxes[bi]
        if coverage_mask[b:t, l:r].min() > 1:
            redundant_bi.append(bi)
            coverage_mask[b:t, l:r] -= 1

    print("Second round removable sample ids:", len(redundant_bi))

    adopted_sample_ids = [each for each in adopted_sample_ids if each not in redundant_bi]

    print("Final adopted sample ids:", len(adopted_sample_ids))

    return dataset, adopted_sample_ids


def pred_func(noisy_contents: List[Tensor], t: Tensor, model: torch.nn.Module, trajs: Tensor):
    pred = model(*noisy_contents, trajs, t)
    return [pred]


def nodesAdjMatToSegs(f_nodes, adj_mat, f_edges, threshold=0.5):
    B = f_nodes.shape[0]
    batch_segs = []
    for b in range(B):
        segs = []
        for r in range(adj_mat.shape[1]):
            for c in range(r + 1, adj_mat.shape[2]):
                if adj_mat[b, r, c] >= threshold:
                    segs.append(f_edges[b, r, c].view(8, 2))
        if len(segs) == 0:
            batch_segs.append(torch.zeros(1, 8, 2, dtype=torch.float32, device=DEVICE))
        else:
            batch_segs.append(torch.stack(segs, dim=0))     # (N_segs, 8, 2)
    return batch_segs


def testGraphWalker(city: str):
    dataset, adopted_sample_ids = prepareDataset(city)

    if city == "Xian":
        dit_path = "Runs/TWDiT/250309_1727_Xian/last.pth"
        vae_path = "Runs/WGVAE_NEW/250309_1310_Xian/last.pth"
    elif city == "Chengdu":
        dit_path = "Runs/TWDiT/250310_0255_Chengdu/last.pth"
        vae_path = "Runs/WGVAE_NEW/250309_2247_Chengdu/last.pth"

    vae = WGVAE_new(N_interp=dataset.N_interp, threshold=0.5).to(DEVICE)
    loadModels(vae_path, vae=vae)
    vae.eval()

    DiT = TWDiT(D_in=dataset.N_interp * 2,
                     N_routes=dataset.N_trajs,
                     L_route=dataset.max_L_route,
                     L_traj=dataset.max_L_traj,
                     d_context=2,
                     n_layers=8,
                     T=500).to(DEVICE)
    loadModels(dit_path, DiT=DiT)

    ddim = DDIM(0.0001, 0.05, 500, DEVICE, "quadratic",
                skip_step=10, data_dim=3)

    # --- Run Model on adopted samples and visualize ---------------------------

    fig = plt.figure(figsize=(10, 10))

    with open(f"./reports/_RecoverCity/GraphWalker_{city}.csv", "w") as f:
        f.write(",".join(titles) + "\n")

        for bi, batch in enumerate(tqdm(dataset)):

            if bi in adopted_sample_ids:
                # Draw input trajectories
                # batch_points = (batch["trajs"] * batch["std_point"] + batch["mean_point"]).view(-1, 2).cpu().numpy()
                # plt.scatter(batch_points[:, 0], batch_points[:, 1], marker=",", s=1, alpha=0.1, color="black")

                # Draw target Road Network
                target = batch["segs"][0][:batch["N_segs"][0]]
                # norm_segs = (target * batch["std_point"][0] + batch["mean_point"][0]).cpu().numpy()
                # for seg in norm_segs:
                #     plt.plot(seg[:, 0], seg[:, 1], linewidth=0.5, color="black")


                # Draw Predicted Road Network
                with torch.no_grad():
                    latent, _ = vae.encode(batch["routes"])
                    latent_noise = torch.randn_like(latent)
                    latent_pred = ddim.diffusionBackward([latent_noise], pred_func, mode="eps", model=DiT, trajs=batch["trajs"])[0]
                    duplicate_segs, sim_mat, uniqueness_mask, unique_seqs = vae.decode(latent_pred)
                batch_scores = reportAllMetrics(unique_seqs, [target])
                f.write(",".join([f"{s[0]}" for s in batch_scores]) + "\n")

                norm_segs = (unique_seqs[0] * batch["std_point"][0] + batch["mean_point"][0]).cpu().numpy()
                for seg in norm_segs:
                    plt.plot(seg[:, 0], seg[:, 1], linewidth=0.5, color="black")

    plt.savefig(f"./reports/_RecoverCity/GraphWalker_Rec_{city}_10.png", dpi=200)


def testGraphusion(city: str):
    dataset, adopted_sample_ids = prepareDataset(city)

    if city == "Xian":
        dit_path = "Runs/Graphusion/250316_0030_Xian/last.pth"
        vae_path = "Runs/GraphusionVAE/250315_2350_Xian/last.pth"
    elif city == "Chengdu":
        dit_path = "Runs/Graphusion/250316_1638_Chengdu/last.pth"
        vae_path = "Runs/GraphusionVAE/250316_1618_Chengdu/last.pth"

    vae = GraphusionVAE(d_node=2, d_edge=16, d_latent=128, d_hidden=256, n_layers=8, n_heads=8).to(DEVICE)
    loadModels(vae_path, vae=vae)
    vae.eval()

    graphusion = Graphusion(D_in=128,
                            L_enc=dataset.max_N_nodes,
                            N_trajs=dataset.N_trajs,
                            L_traj=dataset.max_L_traj,
                            d_context=2,
                            n_layers=6,
                            T=500).to(DEVICE)
    loadModels(dit_path, graphusion=graphusion)
    graphusion.eval()

    ddim = DDIM(0.0001, 0.05, 500, DEVICE, "quadratic",
                skip_step=10, data_dim=3)

    # --- Run Model on adopted samples and visualize ---------------------------

    fig = plt.figure(figsize=(10, 10))

    with open(f"./reports/_RecoverCity/Graphusion_{city}.csv", "w") as f:
        f.write(",".join(titles) + "\n")

        for bi, batch in enumerate(tqdm(dataset)):

            if bi in adopted_sample_ids:
                # Draw input trajectories
                # batch_points = (batch["trajs"] * batch["std_point"] + batch["mean_point"]).view(-1, 2).cpu().numpy()
                # plt.scatter(batch_points[:, 0], batch_points[:, 1], marker=",", s=1, alpha=0.1, color="black")

                # Draw target Road Network
                target = batch["segs"][0][:batch["N_segs"][0]]
                # norm_segs = (target * batch["std_point"][0] + batch["mean_point"][0]).cpu().numpy()
                # for seg in norm_segs:
                #     plt.plot(seg[:, 0], seg[:, 1], linewidth=0.5, color="black")


                # Draw Predicted Road Network
                with torch.no_grad():
                    latent, _ = vae.encode(batch["nodes"], batch["edges"], batch["adj_mat"])
                    latent_noise = torch.randn_like(latent)
                    latent_pred = ddim.diffusionBackward([latent_noise], pred_func, mode="eps",
                                                         model=graphusion, trajs=batch["trajs"])[0]
                    f_nodes, f_edges, pred_adj_mat, pred_degrees = vae.decode(latent_pred)

                pred_segs = nodesAdjMatToSegs(f_nodes, pred_adj_mat, f_edges)
                batch_scores = reportAllMetrics(pred_segs, [target])
                f.write(",".join([f"{s[0]}" for s in batch_scores]) + "\n")

                norm_segs = (pred_segs[0] * batch["std_point"][0] + batch["mean_point"][0]).cpu().numpy()
                for seg in norm_segs:
                    plt.plot(seg[:, 0], seg[:, 1], linewidth=0.5, color="black")

    plt.savefig(f"./reports/_RecoverCity/Graphusion_Rec_{city}.png", dpi=200)


def testTR2RM(city: str):
    dataset, adopted_sample_ids = prepareDataset(city)

    model = AD_Linked_Net(d_in=4, H=256, W=256).to(DEVICE)
    node_extractor = NodeExtractor().to(DEVICE)
    if city == "Xian":
        loadModels("Runs/TR2RM/250315_2235_Xian/last.pth", ADLinkedNet=model)
    else:
        loadModels("Runs/TR2RM/250316_1457_Chengdu/last.pth", ADLinkedNet=model)
    loadModels("Runs/NodeExtractor/241215_1543_Shanghai/last.pth", node_model=node_extractor)

    # --- Run Model on adopted samples and visualize ---------------------------

    fig = plt.figure(figsize=(10, 10))

    with open(f"./reports/_RecoverCity/TR2RM_{city}.csv", "w") as f:
        f.write(",".join(titles) + "\n")

        for bi, batch in enumerate(tqdm(dataset)):

            if bi in adopted_sample_ids:
                # Draw input trajectories
                # batch_points = (batch["trajs"] * batch["std_point"] + batch["mean_point"]).view(-1, 2).cpu().numpy()
                # plt.scatter(batch_points[:, 0], batch_points[:, 1], marker=",", s=1, alpha=0.1, color="black")

                # Draw target Road Network
                target = batch["segs"][0][:batch["N_segs"][0]]
                # norm_segs = (target * batch["std_point"][0] + batch["mean_point"][0]).cpu().numpy()
                # for seg in norm_segs:
                #     plt.plot(seg[:, 0], seg[:, 1], linewidth=0.5, color="black")

                # Draw Predicted Road Network
                with torch.no_grad():
                    pred_heatmap = model(torch.cat([batch["heatmap"], batch["image"]], dim=1))
                    pred_nodemap = node_extractor(batch["target_heatmaps"])
                pred_segs, temp_map = heatmapsToSegments(pred_heatmap, pred_nodemap)
                batch_scores = reportAllMetrics(pred_segs,[target])
                f.write(",".join([f"{s[0]}" for s in batch_scores]) + "\n")

                norm_segs = (pred_segs[0] * batch["std_point"][0] + batch["mean_point"][0]).cpu().numpy()
                for seg in norm_segs:
                    plt.plot(seg[:, 0], seg[:, 1], linewidth=0.5, color="black")

    plt.savefig(f"./reports/_RecoverCity/TR2RM_Rec_{city}.png", dpi=200)


def testDFDRU(city: str):
    dataset, adopted_sample_ids = prepareDataset(city)

    model = DFDRUNet().to(DEVICE)
    node_extractor = NodeExtractor().to(DEVICE)
    if city == "Xian":
        loadModels("Runs/DFDRUNet/250315_2151_Xian/last.pth", DFDRUNet=model)
    else:
        loadModels("Runs/DFDRUNet/250316_1413_Chengdu/last.pth", DFDRUNet=model)
    loadModels("Runs/NodeExtractor/241215_1543_Shanghai/last.pth", node_model=node_extractor)

    # --- Run Model on adopted samples and visualize ---------------------------

    fig = plt.figure(figsize=(10, 10))

    with open(f"./reports/_RecoverCity/DFDRUNet_{city}.csv", "w") as f:
        f.write(",".join(titles) + "\n")

        for bi, batch in enumerate(tqdm(dataset)):

            if bi in adopted_sample_ids:
                # Draw input trajectories
                # batch_points = (batch["trajs"] * batch["std_point"] + batch["mean_point"]).view(-1, 2).cpu().numpy()
                # plt.scatter(batch_points[:, 0], batch_points[:, 1], marker=",", s=1, alpha=0.1, color="black")

                # Draw target Road Network
                target = batch["segs"][0][:batch["N_segs"][0]]
                # norm_segs = (target * batch["std_point"][0] + batch["mean_point"][0]).cpu().numpy()
                # for seg in norm_segs:
                #     plt.plot(seg[:, 0], seg[:, 1], linewidth=0.5, color="black")

                # Draw Predicted Road Network
                with torch.no_grad():
                    pred_heatmap = model(batch["image"], batch["heatmap"])
                    pred_nodemap = node_extractor(batch["target_heatmaps"])
                pred_segs, temp_map = heatmapsToSegments(pred_heatmap, pred_nodemap)
                batch_scores = reportAllMetrics(pred_segs,[target])
                f.write(",".join([f"{s[0]}" for s in batch_scores]) + "\n")

                norm_segs = (pred_segs[0] * batch["std_point"][0] + batch["mean_point"][0]).cpu().numpy()
                for seg in norm_segs:
                    plt.plot(seg[:, 0], seg[:, 1], linewidth=0.5, color="black")

    plt.savefig(f"./reports/_RecoverCity/DFDRUNet_Rec_{city}.png", dpi=200)


def testSmallMap(city: str):
    dataset, adopted_sample_ids = prepareDataset(city)

    stage_1 = UNet2D(n_repeats=2, expansion=2).to(DEVICE)
    stage_2 = UNet2D(n_repeats=2, expansion=2).to(DEVICE)
    node_extractor = NodeExtractor().to(DEVICE)
    if city == "Xian":
        loadModels("Runs/SmallMap/250315_2252_Xian/last.pth", stage_1=stage_1, stage_2=stage_2)
    else:
        loadModels("Runs/SmallMap/250316_1514_Chengdu/last.pth", stage_1=stage_1, stage_2=stage_2)
    loadModels("Runs/NodeExtractor/241215_1543_Shanghai/last.pth", node_model=node_extractor)

    # --- Run Model on adopted samples and visualize ---------------------------

    fig = plt.figure(figsize=(10, 10))

    with open(f"./reports/_RecoverCity/SmallMap_{city}.csv", "w") as f:
        f.write(",".join(titles) + "\n")

        for bi, batch in enumerate(tqdm(dataset)):

            if bi in adopted_sample_ids:
                # Draw input trajectories
                # batch_points = (batch["trajs"] * batch["std_point"] + batch["mean_point"]).view(-1, 2).cpu().numpy()
                # plt.scatter(batch_points[:, 0], batch_points[:, 1], marker=",", s=1, alpha=0.1, color="black")

                # Draw target Road Network
                target = batch["segs"][0][:batch["N_segs"][0]]
                # norm_segs = (target * batch["std_point"][0] + batch["mean_point"][0]).cpu().numpy()
                # for seg in norm_segs:
                #     plt.plot(seg[:, 0], seg[:, 1], linewidth=0.5, color="black")

                # Draw Predicted Road Network
                with torch.no_grad():
                    pred_1 = stage_1(batch["heatmap"])
                    pred_2 = stage_2(pred_1)
                    pred_nodemap = node_extractor(batch["target_heatmaps"])
                pred_segs, temp_map = heatmapsToSegments(pred_2, pred_nodemap)
                batch_scores = reportAllMetrics(pred_segs,[target])
                f.write(",".join([f"{s[0]}" for s in batch_scores]) + "\n")

                norm_segs = (pred_segs[0] * batch["std_point"][0] + batch["mean_point"][0]).cpu().numpy()
                for seg in norm_segs:
                    plt.plot(seg[:, 0], seg[:, 1], linewidth=0.5, color="black")

    plt.savefig(f"./reports/_RecoverCity/SmallMap_Rec_{city}.png", dpi=200)


if __name__ == "__main__":
    # testGraphWalker("Chengdu")
    # testGraphWalker("Xian")

    # testTR2RM("Chengdu")
    # testTR2RM("Xian")

    # testDFDRU("Xian")
    # testDFDRU("Chengdu")

    # testSmallMap("Xian")
    # testSmallMap("Chengdu")

    testGraphusion("Xian")
    testGraphusion("Chengdu")