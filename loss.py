import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CtAslLoss(nn.Module):
    def __init__(self,positive_gamma=None,negative_gamma=None,k=None):
        super(CtAslLoss, self).__init__()
        self.positive_gamma=positive_gamma
        self.negative_gamma=negative_gamma
        self.k=k

    def forward(self, vit_outputs,bert_outputs,labels):
        vit_embeddings=torch.stack(vit_outputs.hidden_states).permute(1,0,2,3)
        bert_embeddings=torch.stack(bert_outputs.hidden_states)[:,:,0,:] #shape=13xMxd
        batch_size,l,n,d=vit_embeddings.shape
        bert_embeddings= bert_embeddings.view(1, *bert_embeddings.shape).expand(batch_size, -1, -1, -1)
        # losses=torch.stack([self.calculate_loss(vit_embeddings[i],bert_embeddings,labels[i]) for i in range(batch_size)])
        losses=self.calculate_loss(vit_embeddings,bert_embeddings,labels)
        # print(losses,"===")
        
        return losses.mean()

    def calculate_loss(self,vit_embeddings,bert_embeddings,labels):
        # ct_distances=[]
        #for every layer vit_embeddings=12*vit_embed
        beta=self.calculate_beta(labels) #labels=bsxM , beta=bsXM
        for i in range(12,vit_embeddings.shape[1]):

            vit_emb=vit_embeddings[:,i]
            bert_emb=bert_embeddings[:,i]
            #E ∈ RNxd and L ∈ RMxd
            E=self.set_E(vit_emb)
            L=self.set_L(bert_emb)
            # print(E.shape,L.shape)
            theta=self.calculate_theta(E,L,labels)#theta=bsxN,

            ct_distances=self.calculate_ct_distance(E,L,theta,beta)

        # ct_distances=torch.tensor(ct_distances).sum(dim=0)
        combined_loss=ct_distances
        return combined_loss

    def set_L(self,bert_emb):
        return bert_emb# first is cls

    def set_E(self,vit_emb):
        return vit_emb[:,1:]#first is cls

    def set_x(self,vit_embeddings):
        return vit_embeddings[:,12,0]

    def calculate_tij(self,p, q, theta, beta):
        #E ∈ R bsxNxd and L ∈ R bsxMxd and theta∈ R bsxN and beta ∈ R bsxM and out tij= bs x N x M
        bs=len(p)
        n=len(p[1])
        m=len(q[1])
        matrix=torch.zeros((bs,n,m)).to(device)
        for i in  range(n):
          for j in range(m):
            denom_temp=torch.mul(torch.exp(self.psi(p[:,i],q[:,j])),beta[:,j])
            num=torch.mul(denom_temp,theta[:,i])
            # sum=sum+denom_temp
            num_expanded = num[:, None, None]
            
            # =============== this was original and has changed 
            # matrix[:,i,j]=num_expanded.squeeze()
            matrix[:,i,j]=num
            # print(num_expanded.squeeze().shape,num_expanded.squeeze())
        #   sum2= sum.view(-1, 1, 1)
        #   result = matrix/sum.view(-1,1,1)
        #   matrix[:, i] = result[:, i]
        return matrix

    def psi(self,ei, lj):
        # e1=bsxd and lj=bsxd
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(ei,lj)

    def calculate_cost_matrix(self,E,L):
        # E ∈ R bsxdxN and L ∈ R bsxdxM and output cost_matrix=bsxNxM
        # print("++++",E.shape,L.shape)
        num=torch.matmul(E.permute(0,2,1)[:,],L.permute(0,2,1)[:,])
        norm=torch.mul(torch.norm(torch.norm(E,dim=1),dim=1),torch.norm(torch.norm(L,dim=1),dim=1))
        cosine_similarity = num / norm.view(-1, 1, 1)
        cost_matrix=torch.sub(torch.ones_like(cosine_similarity),cosine_similarity)
        return cost_matrix

    def calculate_ct_distance(self,E,L,theta,beta):
        # E ∈ RbsxNxd and L ∈ RbsxMxd and theta∈ R bsxN and beta ∈ Rbs xM
        tij=self.calculate_tij(E,L,theta,beta).to(device)
        tji=self.calculate_tij(L,E,beta,theta).to(device)
        cost_matrix=self.calculate_cost_matrix(E.permute(0,2,1),L.permute(0,1,2))
        forward_ct=torch.mul(tij[:,],cost_matrix[:,]).sum(dim=(1,2))
        backward_ct=torch.mul(tji[:,],cost_matrix.permute(0,2,1)[:,]).sum(dim=(1,2))
        # print(forward_ct,"==",backward_ct)
        ct_distance=forward_ct+backward_ct
        return ct_distance

    def calculate_theta(self,E,L,labels):
        #labels=y∈ bsXM,E ∈ R bsxNxd and L ∈ R bsxMxd
        normalized_y=F.normalize(labels, p=1, dim=1)
        o=torch.matmul(L.permute(0,2,1)[:,],normalized_y.view(*normalized_y.shape, 1)[:,]) #o ∈ bsxdx1
        theta=self.topk(torch.matmul(E[:,],o[:,]),self.k) ##1st arg result  ∈ bsxNx1
        
        return theta.view(*theta.shape[:-1])

    def topk(self,x,k):
        #x ∈  bs x N
        _, indices = torch.topk(x,k,dim=1)
        result_tensor = torch.zeros_like(x)
        result_tensor.scatter_(1, indices, 1)
        return result_tensor

    def calculate_beta(self,labels):
        return nn.Softmax(dim=1)(labels.float())

    def calculate_asl(self,L,x,y):
        #L ∈RMxdand x∈dx1
        p=torch.sigmoid(torch.matmul(L,x))
        asl=0
        for i,yi in enumerate(y):
          if(yi==1):
            asl+=((1-p[i])**self.positive_gamma)*torch.log(p[i])
          else:
            asl+=((p[i])**self.negative_gamma)*torch.log(1-p[i])
        return asl

    ##USAGE

    # p = torch.tensor([[0.1, 0.1, 0.3, 0.4],[0.2, 0.2, 0.3, 0.4],[0.1, 0.1, 0.3, 0.3]])
    # q = torch.tensor( [[0.1, 0.1, 0.2, 0.1],[0.2, 0.2, 0.2, 0.1]])
    # theta = torch.tensor([0.4, 0.3, 0.3])
    # beta  = torch.tensor([0.5, 0.5])
    # print(p.size())
    # print(q.size())
    # # Ei ∈ RNxd and L ∈ RMxd
    # ct_distance=ct_distance(p,q,theta,beta)
    # print(ct_distance)