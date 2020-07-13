import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def train_one_epoch(model, loss_fn, device, data_loader, optimizer, epoch):
    model.train()
    for batch_idx, (X, Y) in enumerate(data_loader):
        # X.shape: [seq_len, batch_size, input_size]
        # Y.shape: [seq_len, batch_size]
        # print((X.shape, Y.shape))
        X = X.permute(0, 2, 1).float()
        Y = Y.float()
        # Y = Y.permute(1, 0)

        X, Y = X.to(device), Y.to(device)

        # Zero out gradient, else they will accumulate between epochs
        optimizer.zero_grad()

        # Forward pass
        output = model(X)
        loss = loss_fn(output.squeeze(), Y.reshape(-1))

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()


def test(model, loss_fn, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    val_y_true = []
    val_proba = []
    with torch.no_grad():
        for X, Y in test_loader:
            # X.shape: [seq_len, batch_size, input_size]
            # Y.shape: [seq_len, batch_size]
            X = X.permute(0, 2, 1).float()
            Y = Y.float()
            # X = X.permute(1, 2, 0)
            # Y = Y.permute(1, 0)

            X, Y = X.to(device), Y.to(device)

            output = model(X).squeeze()

            test_loss += loss_fn(output, Y.reshape(-1)
                                 ).item()  # sum up batch loss

            proba = torch.sigmoid(output)
            # output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            pred = (proba > 0.5)

            correct += pred.eq(Y.reshape(-1).bool()).sum().item()

            val_y_true.append(Y.reshape(-1))
            val_proba.append(proba)

    val_y_true = torch.cat(val_y_true)
    val_proba = torch.cat(val_proba)
    test_loss /= (len(test_loader.dataset))

    return val_y_true, val_proba, {'loss': test_loss, 'correct': correct}

def eval(model, loss_fn, device, test_loader):
    val_y_true, val_proba, test_stats = test(model, loss_fn, device, test_loader)
    test_auc = roc_auc_score(val_y_true, val_proba)

    print('\nTest set: Average loss: {:.4f}, AUC: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_stats['loss'], test_auc, test_stats['correct'], len(
            test_loader.dataset),
        100. * test_stats['correct'] / (len(test_loader.dataset))))
