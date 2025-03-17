from .Basics import *
class Deduplicator(nn.Module):
    def __init__(self, d_in: int, threshold: float):
        super().__init__()
        self.d_in = d_in
        self.threshold = threshold

        self.in_proj = nn.Sequential(
            nn.Linear(d_in, d_in * 4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(d_in * 4, d_in * 4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(d_in * 4, d_in * 4),
            nn.LeakyReLU(inplace=True),
            nn.Linear(d_in * 4, d_in * 4)
        )

        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.tau = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, seqs: Float[Tensor, "B L D"]):
        B, L, D = seqs.shape

        # STEP 1. get pair-wise similarity matrix
        q, k = torch.split(self.in_proj(seqs), self.d_in * 2, dim=2)
        sim_mat = torch.cdist(q, k)   # (B, L, L)

        # STEP 2. keep only the similarity scores of previous elements for each element
        mask = torch.tril(torch.ones(L, L, device=seqs.device), diagonal=-1).bool()
        mask = mask.unsqueeze(0).expand(B, -1, -1)  # (B, L, L)
        previous_sim_mat = sim_mat.masked_fill(~mask, -1e9)

        # STEP 3. get the maximum previous similarity score
        # high score means exists previous similar element
        max_previous_sim, _ = previous_sim_mat.max(dim=2)   # (B, L)

        # STEP 4. construct gating, first appear element maps to 1, others map tp 0
        uniqueness_mask = 1 - torch.sigmoid(self.alpha * (max_previous_sim - self.tau)).unsqueeze(-1)  # (B, L, 1)

        # STEP 5. return uniqueness_mask and unnique_seqs
        unique_seqs = []
        for b in range(B):
            unique_seqs.append(seqs[b][uniqueness_mask[b].squeeze(1) > self.threshold].unflatten(-1, (-1, 2)))
        return uniqueness_mask, unique_seqs

def heuristicDeduplication(seqs: Float[Tensor, "B L D"], threshold: float):
    B, L, D = seqs.shape

    # STEP 1. get pair-wise similarity matrix
    sim_mat = torch.cdist(seqs, seqs)  # (B, L, L)

    # STEP 2. keep only the similarity scores of previous elements for each element
    mask = torch.tril(torch.ones(L, L, device=seqs.device), diagonal=-1).bool()
    mask = mask.unsqueeze(0).expand(B, -1, -1)  # (B, L, L)
    previous_sim_mat = sim_mat.masked_fill(~mask, -1e9)

    # STEP 3. get the maximum previous similarity score
    # high score means exists previous similar element
    max_previous_sim, _ = previous_sim_mat.max(dim=2)  # (B, L)

    # STEP 4. construct gating, first appear element maps to 1, others map tp 0
    uniqueness_mask = 1 - torch.sigmoid(max_previous_sim).unsqueeze(-1)  # (B, L, 1)

    # STEP 5. return uniqueness_mask and unnique_seqs
    unique_seqs = []
    for b in range(B):
        unique_seqs.append(seqs[b][uniqueness_mask[b].squeeze(1) > threshold].unflatten(-1, (-1, 2)))
    return uniqueness_mask, unique_seqs

