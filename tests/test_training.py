from torch.utils.data import Dataset
from module6.data import corrupt_mnist
from module6.data import MyDataset
import torch
from module6.model import MyAwesomeModel
import typer

def test_training():
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=1)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    for i, (img, target) in enumerate(train_dataloader):
        img, target = img.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(img)
        assert loss_fn(y_pred, target) != 0
        break
