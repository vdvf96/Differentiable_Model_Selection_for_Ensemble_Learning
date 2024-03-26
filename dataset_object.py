import torch

class Dataset(torch.utils.data.Dataset):
    
    # Create Torch Dataset object.
    def __init__(self, X, Y):

        X = X.reshape((-1, 1, 64, 64))  
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __init__(self, X, Y, s):
        if (s=='rgb'):
            #X = X.reshape((-1, 1, 64, 64))
            self.X = torch.from_numpy(X)
            self.Y = torch.from_numpy(Y)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        X = self.X[index]
        Y = self.Y[index]

        return {'X': X, 'Y': Y}
