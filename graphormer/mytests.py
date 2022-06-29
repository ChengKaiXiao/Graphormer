import torch
from torch import Tensor

t = Tensor([[ 0.0000,  0.0000, 13.0000],
            [ 2.5527,  0.0000, 13.0000],
            [ 0.0000,  2.5527, 13.0000],
            [ 2.5527,  2.5527, 13.0000],
            [ 1.2763,  1.2763, 14.8050],
            [ 3.8290,  1.2763, 14.8050],
            [ 1.2763,  3.8290, 14.8050],
            [ 3.8290,  3.8290, 14.8050],
            [ 0.0000,  0.0000, 16.6100],
            [ 2.5527,  0.0000, 16.6100],
            [ 0.0000,  2.5527, 16.6100],
            [ 2.5527,  2.5527, 16.6100],
            [ 2.5527,  2.5527, 19.6100],
            [ 2.5527,  2.5527, 18.4597]])

print(t)
print(t.size())

cell = Tensor([[[ 5.1053,  0.0000,  0.0000],
         [ 0.0000,  5.1053,  0.0000],
         [ 0.0000,  0.0000, 32.6100]]])
pos = t

cell_offsets = torch.tensor(
            [
                [-1, -1, 0],
                [-1, 0, 0],
                [-1, 1, 0],
                [0, -1, 0],
                [0, 1, 0],
                [1, -1, 0],
                [1, 0, 0],
                [1, 1, 0],
            ],
        ).float()

n_cells = cell_offsets.size(0)
filter_by_tag = True

offsets = torch.matmul(cell_offsets, cell).view(n_cells, 1, 3)
print(offsets)
expand_pos = (pos.unsqueeze(0).expand(n_cells, -1, -1) + offsets).view(
    -1, 3
)
print(expand_pos)
print(expand_pos.size())
expand_pos_relaxed = (
    pos.unsqueeze(0).expand(n_cells, -1, -1) + offsets
).view(-1, 3)
# src_pos = pos[tags > 1] if filter_by_tag else pos

# dist: Tensor = (src_pos.unsqueeze(1) - expand_pos.unsqueeze(0)).norm(dim=-1)
# used_mask = (dist < self.cutoff).any(dim=0) & tags.ne(2).repeat(
#     self.n_cells
# )  # not copy ads
# used_expand_pos = expand_pos[used_mask]
# used_expand_pos_relaxed = expand_pos_relaxed[used_mask]

# used_expand_tags = tags.repeat(self.n_cells)[
#     used_mask
# ]  # original implementation use zeros, need to test
# return dict(
#     pos=torch.cat([pos, used_expand_pos], dim=0),
#     atoms=torch.cat([atoms, atoms.repeat(self.n_cells)[used_mask]]),
#     tags=torch.cat([tags, used_expand_tags]),
#     real_mask=torch.cat(
#         [
#             torch.ones_like(tags, dtype=torch.bool),
#             torch.zeros_like(used_expand_tags, dtype=torch.bool),
#         ]
#     ),
#     deltapos=torch.cat(
#         [pos_relaxed - pos, used_expand_pos_relaxed - used_expand_pos], dim=0
#     ),
#     relaxed_energy=data["relaxed_energy"],
#     )
