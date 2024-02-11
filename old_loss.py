import torch.nn as nn
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CtAslLoss1111(nn.Module):
    def __init__(self,positive_gamma=None,negative_gamma=None,k=None):
        super(CtAslLoss, self).__init__()
        self.positive_gamma=positive_gamma
        self.negative_gamma=negative_gamma
        self.k=k


    def forward(self, vit_outputs,bert_outputs,labels):
        vit_embeddings=torch.stack(vit_outputs.hidden_states).permute(1,0,2,3)
        bert_embeddings=torch.stack(bert_outputs.hidden_states)[:,:,0,:] #shape=13xMxd
        print("shapes: ", vit_embeddings.shape, " bert embeddings shape: ",bert_embeddings.shape, " labels.shape: ", labels.shape)
        batch_size,l,n,d=vit_embeddings.shape
        losses=torch.stack([self.calculate_loss(vit_embeddings[:,i],bert_embeddings,labels[i]) for i in range(batch_size)])
        return losses.mean()


    def calculate_loss(self,vit_embeddings,bert_embeddings,labels):
        ct_distances=[]
        #for every layer vit_embeddings=12*vit_embed
        for vit_emb,bert_emb in zip(vit_embeddings,bert_embeddings):
                #E ∈ RNxd and L ∈ RMxd
                print("vit emb.shape: ",vit_emb.shape," bert_emb.shape: ",bert_emb.shape)
                E=self.set_E(vit_emb)
                L=self.set_L(bert_emb)
                print("E.shape: ",E.shape," L.shape: ",L.shape," labels.shape: ",labels.shape)
                theta=self.calculate_theta(E,L,labels)
                beta=self.calculate_beta(labels)
                ct_distances.append(self.calculate_ct_distance(E,L,theta,beta))
        ct_distances=torch.tensor(ct_distances).to(device)
        x=self.set_x(vit_embeddings)
        asl=self.calculate_asl(L,x,labels)
        combined_loss=torch.add(ct_distances.sum(),asl)
        return combined_loss

    def set_L(self,bert_emb):
        return bert_emb# first is cls

    def set_E(self,vit_emb):
        return vit_emb[1:]#first is cls

    def set_x(self,vit_embeddings):
        return vit_embeddings[12][0]


    def calculate_tij(self,p, q, theta, beta):
        #E ∈ RNxd and L ∈ RMxd and theta∈ RN and beta ∈ RM
        t_matrix = torch.zeros((len(p), len(q)))
        for i, ei in enumerate(p):
            for j, lj in enumerate(q):
                numerator = theta[i] * beta[j] * torch.exp(-self.psi(ei, lj))
                denominator=torch.tensor([beta[k]*torch.exp(-self.psi(ei,each_ele)) for k,each_ele in enumerate(q)]).sum()
                t_matrix[i, j] = numerator / denominator
        return t_matrix

    def psi(self,ei, lj):
        cosine_similarity=torch.matmul(ei,lj)/(torch.norm(ei)*torch.norm(lj))
        return cosine_similarity

    def calculate_cost_matrix(self,E,L):
        # E ∈ RdxN and L ∈ RdxM
        cosine_similarity=torch.matmul(E.t(),L)/(torch.norm(E)*torch.norm(L))
        (n,m)=tuple(cosine_similarity.size())
        cost_matrix=torch.sub(torch.ones(n,m).to(device),cosine_similarity)
        return cost_matrix

    def calculate_ct_distance(self,E,L,theta,beta):
        # E ∈ RNxd and L ∈ RMxd and theta∈ RN and beta ∈ RM
        tij=self.calculate_tij(E,L,theta,beta).to(device)
        tji=self.calculate_tij(L,E,beta,theta).to(device)
        cost_matrix=self.calculate_cost_matrix(E.t(),L.t())
        forward_ct=torch.mul( tij,cost_matrix).sum()
        backward_ct=torch.mul(tji,cost_matrix.t()).sum()
        ct_distance=torch.add(forward_ct,backward_ct)
        return ct_distance

    def calculate_theta(self,E,L,labels):
        #labels=y∈ 1xM,E ∈ RNxd and L ∈ RMxd
        normalized_y=labels/labels.sum()
        o=torch.matmul(L.t(),normalized_y.t()) #o ∈ dx1
        theta=self.topk(torch.matmul(E,o),self.k)
        return theta

    def topk(self,x,k):
        _, indices = torch.topk(x,k)
        result_tensor = torch.zeros_like(x)
        result_tensor[indices] = 1
        return result_tensor

    def calculate_beta(self,labels):
        return nn.Softmax()(labels.float())

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