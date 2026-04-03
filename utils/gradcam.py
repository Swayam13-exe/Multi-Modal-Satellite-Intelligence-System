import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.hook1 = self.target_layer.register_forward_hook(self.save_activation)
        self.hook2 = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.hook1.remove()
        self.hook2.remove()

def generate_gradcam(fusion_model, img_tensor, meta_tensor, target_class_idx=None):
    """
    Generates a Grad-CAM saliency map for the CNN encoder.
    """
    target_layer = fusion_model.cnn_encoder.features[-1]
    
    fusion_model.eval()
    prev_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    
    cam = GradCAM(fusion_model, target_layer)
    
    class_logits, _, _ = fusion_model(img_tensor, meta_tensor)
    
    if target_class_idx is None:
        target_class_idx = torch.argmax(class_logits, dim=1).item()
        
    fusion_model.zero_grad()
    target_score = class_logits[0, target_class_idx]
    target_score.backward()
    
    gradients = cam.gradients
    activations = cam.activations
    
    b, k, u, v = gradients.size()
    alpha = gradients.view(b, k, -1).mean(2)
    weights = alpha.view(b, k, 1, 1)
    
    saliency_map = (weights * activations).sum(1, keepdim=True)
    saliency_map = F.relu(saliency_map)
    saliency_map = F.interpolate(saliency_map, size=(img_tensor.shape[2], img_tensor.shape[3]), mode='bilinear', align_corners=False)
    
    saliency_map_np = saliency_map.squeeze().cpu().detach().numpy()
    if np.max(saliency_map_np) > 0:
        saliency_map_np = saliency_map_np / np.max(saliency_map_np)
        
    cam.remove_hooks()
    torch.set_grad_enabled(prev_grad_state)
    
    return saliency_map_np
