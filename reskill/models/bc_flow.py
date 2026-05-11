import torch

from models.flow_prior import FlowTeacher, FlowStudent, compute_flow_loss, compute_distill_loss

class Flow_BC(object):
    """
    Offline train condition:
        cond = [s0, noisy_a0]
        
    Online sample condition:
        cond = [o, n]
    """
    
    def __init__(
        self,
        cond_dim,
        latent_dim,
        max_action,
        device,
        hidden_dim=256,
        time_dim=16,
        flow_steps=10,
        lr=3e-4,
        distill_coef=1.0,
        grad_clip=None
    ):
        self.teacher = FlowTeacher(
            cond_dim=cond_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            device=device
        ).to(device)
        
        self.student = FlowStudent(
            cond_dim=cond_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            device=device
        ).to(device)
        
        self.actor = self.student
        self.optimizer = torch.optim.Adam(
            list(self.teacher.parameters()) + list(self.student.parameters()),
            lr=lr
        )
        
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.action_dim = latent_dim
        self.max_action = max_action
        self.device = device
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.flow_steps = flow_steps
        self.distill_coef = distill_coef
        self.grad_clip = grad_clip
        
    def _to_tensor(self, x):
        if torch.is_tensor(x):
            return x.to(self.device)
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)
    
    def train(self, cond, target_z, iterations):
        cond = self._to_tensor(cond)
        target_z = self._to_tensor(target_z)
        
        self.teacher.train()
        self.student.train()
        
        metric = {
            "flow_loss": [],
            "distill_loss": [],
            "total_loss": []
        }
        
        for _ in range(iterations):
            bc_flow_loss, teacher_stats = compute_flow_loss(self.teacher, cond, target_z)
            distill_loss, student_stats = compute_distill_loss(self.teacher, self.student, cond, self.flow_steps)
            loss = bc_flow_loss + self.distill_coef * distill_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(self.teacher.parameters()) + list(self.student.parameters()),
                    self.grad_clip
                )
            self.optimizer.step()
            
            metric["flow_loss"].append(bc_flow_loss.item())
            metric["distill_loss"].append(distill_loss.item())
            metric["total_loss"].append(loss.item())
            
        return metric
    
    def sample_z_torch(self, cond):
        """
        cond: [o, n], size: [B, cond_dim]
        """
        batch_size = cond.shape[0]
        
        cond = self._to_tensor(cond)
        noise = torch.randn(batch_size, self.latent_dim, device=cond.device, dtype=cond.dtype)

        self.student.eval()
        with torch.no_grad():
            z = self.student(cond, noise)
            z = z.clamp(-self.max_action, self.max_action)
            
        return z
    
    def save_model(self, dir, id=None):
        suffix = f"_{id}" if id is not None else ""
        torch.save(self.teacher.state_dict(), f"{dir}/teacher{suffix}.pth")
        torch.save(self.student.state_dict(), f"{dir}/student{suffix}.pth")
        
    def load_model(self, dir, id=None):
        suffix = f"_{id}" if id is not None else ""
        self.teacher.load_state_dict(torch.load(f"{dir}/teacher{suffix}.pth", map_location=self.device))
        self.student.load_state_dict(torch.load(f"{dir}/student{suffix}.pth", map_location=self.device))