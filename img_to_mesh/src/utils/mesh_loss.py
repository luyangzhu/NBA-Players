import torch
import torch.nn as nn
import numpy as np


def cotangent(V, F):
    """
    Input:
      V: B x N x 3
      F: B x F  x3
    Outputs:
      C: B x F x 3 list of cotangents corresponding
        angles for triangles, columns correspond to edges 23,31,12

    B x F x 3 x 3
    """
    indices_repeat = torch.stack([F, F, F], dim=2)

    #v1 is the list of first triangles B*F*3, v2 second and v3 third
    v1 = torch.gather(V, 1, indices_repeat[:, :, :, 0].long())
    v2 = torch.gather(V, 1, indices_repeat[:, :, :, 1].long())
    v3 = torch.gather(V, 1, indices_repeat[:, :, :, 2].long())

    l1 = torch.sqrt(((v2 - v3)**2).sum(2)) #distance of edge 2-3 for every face B*F
    l2 = torch.sqrt(((v3 - v1)**2).sum(2))
    l3 = torch.sqrt(((v1 - v2)**2).sum(2))

    # semiperimieters
    sp = (l1 + l2 + l3) * 0.5

    # Heron's formula for area #FIXME why the *2 ? Heron formula is without *2 It's the 0.5 than appears in the (0.5(cotalphaij + cotbetaij))
    A = 2*torch.sqrt( sp * (sp-l1)*(sp-l2)*(sp-l3))

    # Theoreme d Al Kashi : c2 = a2 + b2 - 2ab cos(angle(ab))
    cot23 = (l2**2 + l3**2 - l1**2)
    cot31 = (l1**2 + l3**2 - l2**2)
    cot12 = (l1**2 + l2**2 - l3**2)

    # 2 in batch #proof page 98 http://www.cs.toronto.edu/~jacobson/images/alec-jacobson-thesis-2013-compressed.pdf
    C = torch.stack([cot23, cot31, cot12], 2) / torch.unsqueeze(A, 2) / 4

    return C

class Laplacian(nn.Module):
    def __init__(self, faces):
        super(Laplacian, self).__init__()
        """
        Faces is B x F x 3, cuda torch Variabe.
        Reuse faces.
        """
        self.F = faces.long()
        self.device = faces.device
    
    def compute_laplacian(self, V):
        '''
        Input:
            V: vertices, (B,Vn,3)
        output:
            Lap: cotagent Laplacian matrix, (Bn,Vn,Vn)
        '''
        Bn,Vn = V.shape[0], V.shape[1]
        Fn = self.F.shape[1]
        # 0.5*cotant weight: (Bn,Fn,3), for the last dim, correspond to edge 12,20,01
        C = cotangent(V, self.F)
        ''' build discrete Laplacian-Beltrami operator '''
        # (1,Bn*Fn*3) (0..0,1..1,BS-1...BS-1)
        B_ind = torch.arange(Bn, dtype=torch.long, device=self.device).repeat_interleave(Fn*3).unsqueeze_(0) 
        rows_ind = self.F[:, :, [1, 2, 0]].reshape(1,-1) # (1,Bn*Fn*3)
        cols_ind = self.F[:, :, [2, 0, 1]].reshape(1,-1) # (1,Bn*Fn*3)
        indices = torch.cat([B_ind, rows_ind, cols_ind],dim=0)
        values = C.reshape(-1)
        # build sparse matrix, we need to use uncoalesced property of torch.sparse,
        #  so that we can make sure w_ij = 0.5*(cot_a+cot_b)
        sparse_L = torch.sparse_coo_tensor(indices, values, size=(Bn,Vn,Vn))
        # pytorch does not have good support for sparse, has to transform back to dense. maybe slower for larger mesh
        dense_L = sparse_L.to_dense()
        # when we build sparse, only consider w_ij, need to make sure w_ji = w_ij
        dense_L = dense_L + dense_L.transpose(1,2).contiguous()
        # degree matrix, (Bn,Vn,Vn)
        diag_vals = torch.sum(dense_L,-1)
        M = torch.diag_embed(diag_vals)
        # Laplacian matrix, (Bn,Vn,Vn)
        Lap =  M - dense_L
        return Lap

    def forward(self, V):
        # We don't want to backprop through the computation of the Laplacian
        with torch.no_grad():
            Lap = self.compute_laplacian(V)
        Lx = torch.matmul(Lap, V)
        return Lx

class LaplacianLoss(object):
    """
    Encourages minimal mean curvature shapes.
    """
    def __init__(self, faces, vert, toref=True):
        self.toref = toref
        
        # V x V
        self.laplacian = Laplacian(faces)
        with torch.no_grad():
            tmp = self.laplacian(vert)
            self.curve_gt = tmp
            if not self.toref:
                self.curve_gt = self.curve_gt*0
    
    def __call__(self, verts):
        pred_Lx = self.laplacian(verts)
        loss = torch.norm(pred_Lx-self.curve_gt, dim=-1).mean()
        return loss

def init_edge_gt(vertices, faces):
    sommet_A_source = vertices[faces[:, 0]]
    sommet_B_source = vertices[faces[:, 1]]
    sommet_C_source = vertices[faces[:, 2]]
    target = []
    target.append(np.sqrt( np.sum((sommet_A_source - sommet_B_source) ** 2, axis=1)))
    target.append(np.sqrt( np.sum((sommet_B_source - sommet_C_source) ** 2, axis=1)))
    target.append(np.sqrt( np.sum((sommet_A_source - sommet_C_source) ** 2, axis=1)))
    return target

def compute_edge_loss(points, faces, target):
    score = 0
    sommet_A = points[:,faces[:, 0]]
    sommet_B = points[:,faces[:, 1]]
    sommet_C = points[:,faces[:, 2]]

    score = torch.abs(torch.sqrt(torch.sum((sommet_A - sommet_B) ** 2, dim=2)) / target[0] -1)
    score = score + torch.abs(torch.sqrt(torch.sum((sommet_B - sommet_C) ** 2, dim=2)) / target[1] -1)
    score = score + torch.abs(torch.sqrt(torch.sum((sommet_A - sommet_C) ** 2, dim=2)) / target[2] -1)
    edge_loss = torch.mean(score)
    return edge_loss
