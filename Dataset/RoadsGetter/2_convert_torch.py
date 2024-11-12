# Since the other two scripts uses osmnx, which will be in a separate environment,
# which may not have torch installed. This script will complete the last step of the dataset preparation.
# Convert and save data to torch tensors, which will be loaded by Dataset Cache Generator (C++).
import numpy as np
import torch
import io

cities = ["Jilin_City", "Shanghai", "Paris", "Las_Vegas", "Tokyo"]

for city in cities:
    print(f"Processing {city}")

    nodes, edges, degrees = np.load(f"{city}_street_network.npz").values()

    nodes = torch.tensor(nodes, dtype=torch.float32, device="cuda")
    edges = torch.tensor(edges, dtype=torch.int64, device="cuda")
    degrees = torch.tensor(degrees, dtype=torch.int64, device="cuda")

    io_stream = io.BytesIO()
    torch.save([nodes, edges, degrees], io_stream, _use_new_zipfile_serialization=True)
    with open(f"{city}_street_network.pt", "wb") as f:
        f.write(io_stream.getbuffer())