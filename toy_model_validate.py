import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

model = torch.load("toy.model")
model.eval()

mask_token = 1

mask = np.ones(10)
mask[:5] = 0
np.random.shuffle(mask)

src = torch.LongTensor([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
src_mask = torch.from_numpy(mask).bool().unsqueeze(0)
masked_src = src.masked_fill(src_mask == 0, mask_token)

print(src, masked_src, src_mask)

output = model.forward(masked_src, src_mask)
prob = F.softmax(model.generator(output), dim=-1)


print(src)
print(torch.argmax(prob, dim=-1))
