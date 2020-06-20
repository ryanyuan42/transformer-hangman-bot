class Batch:
    def __init__(self, src, tgt, mask_token=None, pad_token=0):
        self.src = src
        self.tgt = tgt
        self.src_mask = ((src != mask_token) & (src != pad_token)).unsqueeze(-2)
        self.randomly_mask = (src != mask_token)
