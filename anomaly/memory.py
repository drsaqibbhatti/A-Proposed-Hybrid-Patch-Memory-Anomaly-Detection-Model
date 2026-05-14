from typing import Optional

import torch
from tqdm import tqdm


def _limit_patches(patches: torch.Tensor, max_patches: Optional[int], seed: int = 42) -> torch.Tensor:
    if max_patches is None or max_patches <= 0 or patches.shape[0] <= max_patches:
        return patches
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    idx = torch.randperm(patches.shape[0], generator=gen)[:max_patches]
    return patches[idx]


@torch.no_grad()
def collect_patch_embeddings(
    model,
    dataloader,
    device: torch.device,
    max_train_patches: int = 200000,
    seed: int = 42,
) -> torch.Tensor:
    """Extract all normal patch embeddings and cap them to a manageable candidate set."""
    model.eval()
    chunks = []
    total = 0
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    for batch in tqdm(dataloader, desc="Extracting normal patch embeddings"):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device, non_blocking=True)
        emb = model.extract_embedding(images)
        patches = emb.permute(0, 2, 3, 1).reshape(-1, emb.shape[1]).detach().float().cpu()
        chunks.append(patches)
        total += patches.shape[0]
        if max_train_patches > 0 and total > max_train_patches * 2:
            all_patches = torch.cat(chunks, dim=0)
            idx = torch.randperm(all_patches.shape[0], generator=gen)[:max_train_patches]
            chunks = [all_patches[idx]]
            total = max_train_patches

    if not chunks:
        raise RuntimeError("No embeddings were extracted. Is the training directory empty?")
    patches = torch.cat(chunks, dim=0)
    patches = _limit_patches(patches, max_train_patches, seed=seed)
    return patches.contiguous()


@torch.no_grad()
def kcenter_greedy(
    patches: torch.Tensor,
    num_samples: int,
    device: torch.device,
    seed: int = 42,
) -> torch.Tensor:
    """Approximate k-center greedy coreset selection for PatchCore memory."""
    n = patches.shape[0]
    if num_samples >= n:
        return patches
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    first = int(torch.randint(0, n, (1,), generator=gen).item())

    feats = patches.to(device=device, dtype=torch.float32)
    feats = torch.nn.functional.normalize(feats, dim=1, eps=1e-6)
    min_dist = torch.full((n,), float("inf"), device=device)
    selected_idx = torch.empty((num_samples,), dtype=torch.long, device=device)
    current = first

    for i in tqdm(range(num_samples), desc="Selecting k-center coreset"):
        selected_idx[i] = current
        center = feats[current: current + 1]
        dist = torch.cdist(feats, center, p=2).squeeze(1)
        min_dist = torch.minimum(min_dist, dist)
        current = int(torch.argmax(min_dist).item())

    return patches[selected_idx.cpu()].contiguous()


@torch.no_grad()
def build_memory_bank(
    patches: torch.Tensor,
    coreset_ratio: float = 0.05,
    max_memory_patches: int = 20000,
    method: str = "greedy",
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> torch.Tensor:
    """Create representative memory bank from normal patch embeddings."""
    if patches.ndim != 2:
        raise ValueError(f"Expected patches as NxD, got {tuple(patches.shape)}")
    n = patches.shape[0]
    target = int(max(1, round(n * float(coreset_ratio))))
    if max_memory_patches and max_memory_patches > 0:
        target = min(target, int(max_memory_patches))
    target = min(target, n)

    method = method.lower()
    if method == "random" or target >= n:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        idx = torch.randperm(n, generator=gen)[:target]
        memory = patches[idx]
    elif method in ("greedy", "kcenter", "k-center"):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        memory = kcenter_greedy(patches, target, device=device, seed=seed)
    else:
        raise ValueError("method must be 'greedy' or 'random'")
    return torch.nn.functional.normalize(memory.float(), dim=1, eps=1e-6).cpu().contiguous()


@torch.no_grad()
def nearest_neighbor_distance(
    query: torch.Tensor,
    memory: torch.Tensor,
    chunk_size: int = 2048,
    memory_chunk_size: int = 20000,
) -> torch.Tensor:
    """Minimum Euclidean distance from each query patch to the memory bank."""
    if query.ndim != 2 or memory.ndim != 2:
        raise ValueError("query and memory must be 2D tensors")
    if query.shape[1] != memory.shape[1]:
        raise ValueError(f"Feature dim mismatch: query={query.shape[1]} memory={memory.shape[1]}")

    query = torch.nn.functional.normalize(query.float(), dim=1, eps=1e-6)
    memory = torch.nn.functional.normalize(memory.float(), dim=1, eps=1e-6).to(query.device)
    out = []
    for q_start in range(0, query.shape[0], chunk_size):
        q = query[q_start: q_start + chunk_size]
        best = torch.full((q.shape[0],), float("inf"), device=query.device)
        for m_start in range(0, memory.shape[0], memory_chunk_size):
            m = memory[m_start: m_start + memory_chunk_size]
            dist = torch.cdist(q, m, p=2).min(dim=1).values
            best = torch.minimum(best, dist)
        out.append(best)
    return torch.cat(out, dim=0)
