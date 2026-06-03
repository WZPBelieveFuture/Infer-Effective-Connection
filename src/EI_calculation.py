import numpy as np
import torch
from torch import nn
from torch.autograd.functional import jacobian
def approx_ei(input_size, output_size, sigmas_matrix, func, num_samples, L, easy=True, device=None):
    rho = 1/(2*L)**input_size # the density of X even distribution
    #the first term of EI, the entropy of a gaussian distribution
    #sigmas_matrix_np=sigmas_matrix.cpu() if use_cuda else sigmas_matrix
    dett = 1.0 
    if easy:
        dd = torch.diag(sigmas_matrix)
        dett = torch.log(dd).sum()
    else:
        dett = torch.log(torch.linalg.det(sigmas_matrix))
    term1 = - (1 + 1 * np.log(2*np.pi) + dett/output_size) / 2 
    
    #sampling x on the space [-L,L]^n, n is the number of samples
    xx = L*2*(torch.rand(num_samples, input_size, device=sigmas_matrix.device)-1/2)
    
    dets = 0
    logdets = 0
    
    #iterate all samples of x
    for i in range(xx.size()[0]):
        jac = jacobian(func, xx[i,:]) #use pytorch's jacobian function to obtain jacobian matrix
        det = torch.abs(torch.det(jac)) #calculate the determinate of the jacobian matrix
        dets += det.item()
        if det != 0:
            logdets += torch.log(det).item() #log jacobian
        else:
            #if det==0 then, it becomes a gaussian integration
            logdet = -(output_size+output_size*np.log(2*np.pi)+dett)
            logdets += logdet.item()
    
    int_jacobian = logdets / xx.size()[0] #take average of log jacobian
    
    term2 = int_jacobian / output_size # derive the 2nd term
    
    if dets==0:
        term2 = - term1
    EI = max(term1 + term2 + torch.log(torch.tensor(L, dtype=torch.float)), 0)
    if torch.is_tensor(EI):
        EI = EI.item()
    eff = -EI / np.log(rho)
    d_EI = EI
    return d_EI, eff, term1.item(), term2, -np.log(rho)
