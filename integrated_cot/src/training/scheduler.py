import torch
import numpy as np
from collections import defaultdict


class CurriculumScheduler:
    def __init__(self, config, domains):
        self.config = config
        self.domains = domains
        
        
        self.ema_rate = config['curriculum']['ema_rate']
        self.plateau_patience = config['curriculum']['plateau_patience']
        
        # thresholds for curriculum logic
        self.gamma = config['curriculum']['gamma'] # gap threshold for medium
        self.tau = config['curriculum']['tau'] # temperature for sigmoid
        self.gamma_H = config['curriculum']['gamma_H'] # gap threshold for hard
        self.epsilon_plat = config['curriculum']['epsilon_plat'] # plateau detection
        self.delta_rise = config['curriculum']['delta_rise'] # loss rise detection
        
        # hard stage budget parameters
        self.hard_step_up = config['curriculum']['hard_step_up']
        self.hard_step_down = config['curriculum']['hard_step_down']
        self.hard_budget_max = config['curriculum']['hard_budget_max']
        
        # curriculum phase transition epochs
        self.ramp_kappa = config['curriculum']['ramp_kappa']
        self.easy_warmup_epochs = config['curriculum']['easy_warmup_epochs']
        self.hard_transition_epoch = config['curriculum']['hard_transition_epoch']  #
        
        # ema loss trackers per domain
        self.m_easy = defaultdict(lambda: 1e-3)
        self.m_med = defaultdict(lambda: 1e-3)
        self.m_global = 1e-4
        self.m_global_history = []
        
        self.lambda_H = config['curriculum']['hard_budget_init'] # initial hard budget
        self.current_epoch = 0

        
    def compute_ramp_factor(self, epoch):
        # slowly increases medium probability after warmup
        if epoch <= self.easy_warmup_epochs:
            return 0.0
        return min(1.0, (epoch - self.easy_warmup_epochs) / self.ramp_kappa)
    
    def compute_gap(self, domain):
        # calculates the normalized loss gap between easy and medium stages
        easy_loss = self.m_easy[domain]
        med_loss = self.m_med[domain]
        
        if easy_loss < 1e-8:
            return 0.0
        
        gap = (easy_loss - med_loss) / (easy_loss + 1e-8)
        return gap
    
    def compute_medium_probability(self, domain, epoch):
        # calculates probability of sampling 'medium' based on gap
        beta_e = self.compute_ramp_factor(epoch)
        g_d = self.compute_gap(domain)
        
        logit = (g_d - self.gamma) / self.tau # high gap -> high probability
        prob = beta_e * torch.sigmoid(torch.tensor(logit)).item()
        
        return prob
    
    def assign_stage(self, domain, epoch):
        
        easy_warmup = self.easy_warmup_epochs
        hard_transition = self.hard_transition_epoch
        
        # phase 1: easy only warmup
        if epoch < easy_warmup:
            return "easy"
        
        # phase 2: easy + medium (adaptive sampling)
        elif epoch < hard_transition:
            p_med = self.compute_medium_probability(domain, epoch)
            if np.random.rand() < p_med:
                return "medium"
            else:
                return "easy"
        
        # phase 3: easy + medium + hard
        else:

            # first, check if we sample from the hard budget
            if np.random.rand() < self.lambda_H:
                return "hard"
            
            # if not hard, sample from easy/medium adaptively
            p_med = self.compute_medium_probability(domain, epoch)
            if np.random.rand() < p_med:
                return "medium"
            else:
                return "easy"
    
    def update_ema(self, domain, stage, loss_value):
        # update the exponential moving average of loss for a domain/stage
        if stage == "easy":
            self.m_easy[domain] = (1 - self.ema_rate) * self.m_easy[domain] + self.ema_rate * loss_value
        elif stage == "medium":
            self.m_med[domain] = (1 - self.ema_rate) * self.m_med[domain] + self.ema_rate * loss_value
    
    def update_global_ema(self, global_loss):
        # update the global loss ema
        self.m_global = (1 - self.ema_rate) * self.m_global + self.ema_rate * global_loss
        self.m_global_history.append(self.m_global)
    
    def check_plateau(self):
        # check if the global loss has stopped improving
        if len(self.m_global_history) < self.plateau_patience:
            return False
        
        recent_losses = self.m_global_history[-self.plateau_patience:]
        deltas = [abs(recent_losses[i] - recent_losses[i-1]) for i in range(1, len(recent_losses))]
        
        return all(delta <= self.epsilon_plat for delta in deltas)
    
    def check_loss_rise(self):
        # check if the global loss is increasing (training instability)
        if len(self.m_global_history) < 2:
            return False
        
        delta = self.m_global_history[-1] - self.m_global_history[-2]
        return delta >= self.delta_rise
    
    def compute_median_gap(self):
        # get the median gap across all domains
        gaps = [self.compute_gap(d) for d in self.domains]
        return np.median(gaps) if gaps else 0.0
    
    def update_hard_budget(self):
        
        # adaptively update the hard sampling budget (lambda_h)

        plateau = self.check_plateau()
        median_gap = self.compute_median_gap()
        # if loss plateaus and gap is high, increase hard samples
        if plateau and median_gap >= self.gamma_H:
            self.lambda_H = min(self.lambda_H + self.hard_step_up, self.hard_budget_max)
        # if loss rises, decrease hard samples

        elif self.check_loss_rise():
            self.lambda_H = max(0.0, (1 - self.hard_step_down) * self.lambda_H)
    
    
    def step(self, epoch, epoch_losses):
        # main step function called by the trainer at the end of each epoch
        self.current_epoch = epoch
        # update ema losses for each domain
        for domain, losses in epoch_losses.items():

            easy_losses = [v for k, v in losses.items() if 'easy' in k and 'loss' in k]
            medium_losses = [v for k, v in losses.items() if 'medium' in k and 'loss' in k]
            
            if easy_losses:
                avg_easy_loss = np.mean(easy_losses)
                self.update_ema(domain, 'easy', avg_easy_loss)
            
            if medium_losses:
                avg_medium_loss = np.mean(medium_losses)
                self.update_ema(domain, 'medium', avg_medium_loss)
        
        all_losses = []
        for domain_losses in epoch_losses.values():
            valid_losses = [v for v in domain_losses.values() if isinstance(v, (int, float))]
            all_losses.extend(valid_losses)
        
        if all_losses:
            global_loss = np.mean(all_losses)
            self.update_global_ema(global_loss)
        # finally, update the hard budget based on new ema values
        self.update_hard_budget()
    
    def get_stage_proportions(self):
        return {
            'hard': self.lambda_H
        }
    
    def get_metrics(self):
        gaps = {d: self.compute_gap(d) for d in self.domains}
        
        return {
            'lambda_H': self.lambda_H,
            'median_gap': self.compute_median_gap(),
            'plateau': self.check_plateau(),
            'domain_gaps': gaps,
            'm_global': self.m_global
        }