from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss(reduction='none')

    def vali(self, vali_data, vali_loader, criterion):
        total_losses = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, _ in vali_loader:
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x, None, None, None)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                
                loss = torch.mean(criterion(outputs, batch_x)).item()
                total_losses.append(loss)
        
        self.model.train()
        return np.mean(total_losses)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            
            for batch_x, _ in train_loader:
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                
                outputs = self.model(batch_x, None, None, None)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                
                loss = torch.mean(criterion(outputs, batch_x))
                train_loss.append(loss.item())
                
                loss.backward()
                model_optim.step()

            train_loss = np.mean(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                break
            
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        
        if test:
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduction='none')

        train_energy = self._compute_energy(train_loader)
        test_energy, test_labels = self._compute_energy(test_loader, with_labels=True)

        combined_energy = np.concatenate([train_energy, test_energy])
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels).reshape(-1)
        gt = test_labels.astype(int)

        gt, pred = adjustment(gt, pred)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(gt, pred, average='binary')

        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F-score: {f_score:.4f}")

        with open("result_anomaly_detection.txt", 'a') as f:
            f.write(f"{setting}\n")
            f.write(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F-score: {f_score:.4f}\n\n")

        return

    def _compute_energy(self, dataloader, with_labels=False):
        energies = []
        labels = []

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                energy = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                energies.append(energy.cpu().numpy())
                
                if with_labels:
                    labels.append(batch_y)

        return (np.concatenate(energies), 
                np.concatenate(labels) if with_labels else None)