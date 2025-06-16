import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', 
                 seasonal_patterns=None, percent=10):
        """
        Dataset for ETT Hourly data

        Args:
            root_path: Root directory of dataset
            flag: 'train', 'val', or 'test'
            size: [seq_len, label_len, pred_len], default None sets to default sizes
            features: 'S'(single), 'M'(multiple), or 'MS'(multi with target)
            data_path: CSV filename
            target: target feature column name
            scale: whether to apply StandardScaler
            timeenc: encoding type of time features, 0 or 1
            freq: frequency string, e.g., 'h' for hourly
            percent: percentage of training data to use
        """
        if size is None:
            self.seq_len = 24 * 4 * 4  # default 384
            self.label_len = 24 * 4    # default 96
            self.pred_len = 24 * 4     # default 96
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val', 'test'], "flag must be 'train', 'val' or 'test'"
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        # Load raw CSV data
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Define data splits by index according to set_type
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Adjust border2 for training subset percentage
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        # Select columns based on features type
        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]  # exclude date column
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError("Invalid features flag.")

        # Scale data if required
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Extract time features
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            data_stamp = df_stamp.drop(columns=['date']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError("Unsupported timeenc value")

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', 
                 seasonal_patterns=None, percent=10):
        """
        Dataset for ETT Minute-level data
        """
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError("Invalid features flag.")

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            df_stamp['minute'] = df_stamp.date.dt.minute
            # Map minute to blocks of 15 minutes
            df_stamp['minute'] = df_stamp['minute'].map(lambda x: x // 15)
            data_stamp = df_stamp.drop(columns=['date']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError("Unsupported timeenc value")

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class MSLSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        """
        Dataset class for MSL segmentation
        
        Args:
            root_path: Directory with train/test numpy files
            win_size: window size
            step: stride for sliding window
            flag: 'train', 'val', 'test', or 'all'
        """
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        self.train = self.scaler.transform(data)

        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.val = self.test  # validation set uses test data as per original

        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))

        print("MSL test data shape:", self.test.shape)
        print("MSL train data shape:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:  # 'all' or other
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        idx = index * self.step
        if self.flag == "train":
            return np.float32(self.train[idx:idx + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == "val":
            return np.float32(self.val[idx:idx + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == "test":
            return np.float32(self.test[idx:idx + self.win_size]), np.float32(self.test_labels[idx:idx + self.win_size])
        else:
            base = (index // self.step) * self.win_size
            return (np.float32(self.test[base:base + self.win_size]),
                    np.float32(self.test_labels[base:base + self.win_size]))


class SMAPSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        """
        Dataset class for SMAP segmentation
        
        Args same as MSLSegLoader
        """
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        self.train = self.scaler.transform(data)

        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.val = self.test

        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))

        print("SMAP test data shape:", self.test.shape)
        print("SMAP train data shape:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        idx = index * self.step
        if self.flag == "train":
            return np.float32(self.train[idx:idx + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == "val":
            return np.float32(self.val[idx:idx + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == "test":
            return np.float32(self.test[idx:idx + self.win_size]), np.float32(self.test_labels[idx:idx + self.win_size])
        else:
            base = (index // self.step) * self.win_size
            return (np.float32(self.test[base:base + self.win_size]),
                    np.float32(self.test_labels[base:base + self.win_size]))