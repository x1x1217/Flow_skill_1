import torch

from reskill.models.flow_prior import FlowTeacher, FlowStudent, compute_flow_loss, compute_distill_loss, compute_flow_z, compute_flow_z_guided

class Flow_BC(object):
    """
    Offline train condition:
        cond = [s0, noisy_a0]
        
    Online sample condition:
        cond = [o, n]
        
    latent: z, size: [B, latent_dim]
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
        use_student=True,
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
        
        self.actor = self.student if use_student else self.teacher
        self.use_student = use_student
        trainable_params = list(self.teacher.parameters())
        if self.use_student:
            trainable_params += list(self.student.parameters())
        self.optimizer = torch.optim.Adam(trainable_params, lr=lr)
        
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
    
    def train(self, cond, target_z, iterations, sample_weight=None):
        cond = self._to_tensor(cond)
        target_z = self._to_tensor(target_z)
        sample_weight = None if sample_weight is None else self._to_tensor(sample_weight).view(-1)
        
        self.teacher.train()
        if self.use_student:
            self.student.train()
        
        metric = {
            "flow_loss": [],
            "distill_loss": [],
            "total_loss": []
        }
        
        for _ in range(iterations):
            bc_flow_loss, teacher_stats = compute_flow_loss(
                self.teacher,
                cond,
                target_z,
                sample_weight=sample_weight,
            )
            if self.use_student:
                distill_loss, student_stats = compute_distill_loss(
                    self.teacher,
                    self.student,
                    cond,
                    self.flow_steps,
                    self.max_action
                )
                loss = bc_flow_loss + self.distill_coef * distill_loss
            else:
                distill_loss = torch.zeros((), device=cond.device, dtype=bc_flow_loss.dtype)
                loss = bc_flow_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], self.grad_clip)
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

        with torch.no_grad():
            if self.use_student:
                self.student.eval()
                z = self.student(cond, noise)
                z = z.clamp(-self.max_action, self.max_action)
            else:
                self.teacher.eval()
                z = compute_flow_z(self.teacher, cond, noise, self.flow_steps, self.max_action)

        return z
    
    def sample_z_guided_torch(self, cond, q_fn, n_obs, guidance_scale=0.0, grad_clip=0.0):
        """
        cond: [o, n], size: [B, cond_dim]
        q_fn: latent-level Q function with input [o, z]
        """
        batch_size = cond.shape[0]
        
        cond = self._to_tensor(cond)
        noise = torch.randn(batch_size, self.latent_dim, device=cond.device, dtype=cond.dtype)
        
        if self.use_student:
            raise NotImplementedError("Flow guidance is only implemented for teacher Euler sampling.")
        
        self.teacher.eval()
        z = compute_flow_z_guided(
            self.teacher,
            cond,
            noise,
            self.flow_steps,
            self.max_action,
            q_fn,
            n_obs,
            guidance_scale=guidance_scale,
            grad_clip=grad_clip,
        )
        
        return z

    def save_model(self, dir, id=None):
        suffix = f"_{id}" if id is not None else ""
        torch.save(self.teacher.state_dict(), f"{dir}/teacher{suffix}.pth")
        torch.save(self.student.state_dict(), f"{dir}/student{suffix}.pth")
        
    def load_model(self, dir, id=None):
        suffix = f"_{id}" if id is not None else ""
        self.teacher.load_state_dict(torch.load(f"{dir}/teacher{suffix}.pth", map_location=self.device))
        self.student.load_state_dict(torch.load(f"{dir}/student{suffix}.pth", map_location=self.device))
