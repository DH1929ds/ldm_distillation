import torch
import torch.nn as nn
import torch.nn.functional as F
from funcs import print_gpu_memory_usage
from gpu_log import GPUMonitor


class distillation_DDPM_trainer(nn.Module):
  def __init__(self, T_model, S_model, T_sampler, S_sampler, distill_features = False):
    
    super().__init__()
    
    self.T_model = T_model
    self.S_model = S_model
    self.T_sampler = T_sampler
    self.S_sampler = S_sampler
    self.distill_features = distill_features
        
  def forward(self, x_t, c, t, cfg_scale=1, loss_weight = 0.1):
        """
        Perform the forward pass for knowledge distillation.
        """
        ############################### TODO ###########################
        if self.distill_features :
            # Teacher model forward pass (in evaluation mode)
            with torch.no_grad():
                #teacher_output, teacher_features = self.T_model.forward_features(x_t, t)
                x_prev, pred_x0, T_output, T_features = self.T_sampler.cache_step(x_t, c, t, t,
                                                                      use_original_steps = True, 
                                                                      unconditional_guidance_scale = cfg_scale,
                                                                      is_feature=self.distill_features)

            
            #student_output, student_features = self.S_model.forward_features(x_t, t)
            S_output, S_features = self.S_model.apply_model(x_t, t, c, is_feature=self.distill_features)
            
            output_loss = F.mse_loss(S_output, T_output, reduction='mean')
            
            feature_loss = 0
            for student_feature, teacher_feature in zip(S_features, T_features):
                teacher_feature = teacher_feature
                feature_loss += F.mse_loss(student_feature, teacher_feature, reduction='mean')
                
            total_loss = output_loss + loss_weight * feature_loss / len(S_features)
            
        ############################### TODO ###########################       
        
        else:
            # Teacher model forward pass (in evaluation mode)
            with torch.no_grad():
                #print_gpu_memory_usage("Before Teacher sampler cache step")
                x_prev, pred_x0, T_output = self.T_sampler.cache_step(
                    x_t, c, t, t,
                    use_original_steps=True,
                    unconditional_guidance_scale=cfg_scale
                )
                #print_gpu_memory_usage("After Teacher sampler cache step")

            S_output = self.S_model.apply_model(x_t, t, c)
            
            output_loss = F.mse_loss(T_output, S_output, reduction='mean')
            total_loss = output_loss
            

        return output_loss, total_loss, x_prev