# +
import torch
import torch.nn as nn
import torch.nn.functional as F

class categorical_crossentropy_2d:
    def __init__(self, weights=None,mask=False):
        self.weights = weights
        self.mask = mask
        #self.eps = torch.finfo(torch.float32).eps
        
    def loss(self,y_pred_logits,y_true):

        log_p = F.log_softmax(y_pred_logits, dim=1)
        loss_map = -y_true * log_p
        epsilon = torch.finfo(torch.float32).eps

        if self.mask:
          
            #Loss assessment: W * y_true * (-log_p)
            weighted_loss = (self.weights[0] * loss_map[:, 0, :] + 
                             self.weights[1] * loss_map[:, 1, :] + 
                             self.weights[2] * loss_map[:, 2, :])
  
           
            #Weight assessment: W * y_true
            weighted_targets = (self.weights[0] * y_true[:, 0, :] + 
                                self.weights[1] * y_true[:, 1, :] + 
            
                             self.weights[2] * y_true[:, 2, :])
            
            #Sum and normalization
            loss_sum = torch.sum(weighted_loss)
            weight_sum = torch.sum(weighted_targets)
            
        
            #Normalizes by the total sum of the weights (including epsilon for safety)
            return loss_sum / (weight_sum + epsilon) # Usando 'epsilon' em vez de 'self.eps'
        else:
            loss_sum = torch.sum(loss_map, dim=(1, 2)) 
            prob_sum = torch.sum(y_true, dim=(1, 2)) # Conta o número total de sítios verdadeiros
            
        
            #epsilon = torch.finfo(torch.float32).eps 
            loss_per_batch = loss_sum / (prob_sum + epsilon)
            
            #Returns the average loss per batch.
            return torch.mean(loss_per_batch)
        

class binary_crossentropy_2d:
    def __init__(self):
        self.eps = torch.finfo(torch.float32).eps
        
    def loss(self,y_pred,y_true):
        loss = torch.mean(y_true*torch.log(y_pred+self.eps) + (1-y_true)*torch.log(1-y_pred+self.eps))
        return -loss
    


# -

class kl_div_2d:
    def __init__(self,temp=1):
        self.eps = torch.finfo(torch.float32).eps
        self.temp = temp
        
    def loss(self,student_logits, teacher_logits):
     
        teacher_log_probs = F.log_softmax(teacher_logits / self.temp, dim=1)

        student_log_probs = F.log_softmax(student_logits / self.temp, dim=1)

        prob_teacher = torch.exp(teacher_log_probs)

        kl_loss_map = prob_teacher * (teacher_log_probs - student_log_probs)

        kl_loss = torch.mean(torch.sum(kl_loss_map, dim=(1, 2))) * self.temp**2
        
        return kl_loss
    
class SpliceDistillationLoss(nn.Module):
    #Combining hard target with soft target
    def __init__(self, T=5.0, alpha=0.9, class_weights=None, **kwargs): # ADICIONADO class_weights
   
        super(SpliceDistillationLoss, self).__init__()
        
        self.T = T
        self.alpha = alpha
        
        #Passing weights and mask=True if weights is provided (for class weighting)
        mask = class_weights is not None
        self.hard_loss_fn = categorical_crossentropy_2d(weights=class_weights, mask=mask).loss 
        
        #Soft loss, using KL divergence 
        self.soft_loss_fn = kl_div_2d(temp=self.T).loss 

    def forward(self, student_logits, teacher_logits, hard_targets):
        #The student and professor should return LOGITS
        
        loss_soft = self.soft_loss_fn(student_logits, teacher_logits)
        loss_hard = self.hard_loss_fn(student_logits, hard_targets) 
        loss_total = self.alpha * loss_soft + (1.0 - self.alpha) * loss_hard
        
  
        return loss_total