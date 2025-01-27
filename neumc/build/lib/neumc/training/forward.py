import torch

import neumc


class ForwardTrainer:
    def __init__(self, cfgs, batch_size, action, shuffle=True, torch_device='cuda'):
        self.dataset = torch.utils.data.TensorDataset(cfgs)

        self.dataloader = torch.utils.data.DataLoader(self.dataset, shuffle=shuffle,
                                                      batch_size=batch_size, drop_last=True)
        self.torch_device = torch_device
        self.action = action
        self.batch_size = batch_size

    def train(self, layers, prior, optimizer, n_epochs, n_batches=1):
        for i, (phi,) in enumerate(self.dataloader):
            if i >= n_epochs:
                break
            optimizer.zero_grad()
            for j in range(n_batches):
                phi = phi.to(self.torch_device)
                z, log_q_phi = neumc.nf.flow.reverse_apply_layers(layers, phi,
                                                                  torch.zeros((self.batch_size,),
                                                                              device=phi.device)
                                                                  )
                prob_z = prior.log_prob(z)
                log_q_phi = prob_z - log_q_phi
                loss = -log_q_phi.mean()
                loss /= n_batches
                loss.backward()

            optimizer.step()
        with torch.no_grad():
            log_prob_p = -self.action(phi)
            print(torch.mean(log_q_phi - log_prob_p))
