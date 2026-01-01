import torch
import torch.nn as nn

class NTXent(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z1, z2):
        self.B = z1.shape[0]
        z = torch.cat([self.l2norm(z1),self.l2norm(z2)],dim=0)
        
        S, positives = self.similarityMatrix(z)

    def l2norm(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=1)

    def similarityMatrix(self,z):
        S = z @ z.T
        rows = [row for row in range(2*self.B)]
        cols = []
        for row in rows:
            if row <= (self.B-1):
                cols.append(row+self.B)
            elif row > (self.B-1):
                cols.append(row-self.B)
        positives = S[rows,cols]

        return S, positives
    
    def masking(self, S):
        mask = torch.eye(2*self.B, dtype=torch.bool).to(S.device)
        S = S.masked_fill_(mask=mask, value=float('-inf'))
        
        return S


