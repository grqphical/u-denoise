import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from ..model import UNet
from ..data import RawImageDataset
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model, loader, loss_function):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_function(pred, y).item() * X.size(0)
    return total_loss / len(loader.dataset)


def train_model(output_file: Path = Path("model/udenoise.pkl"), epochs: int = 10):
    dataset = RawImageDataset()

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size  # remainder avoids rounding gaps

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),  # reproducible splits
    )

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    unet = UNet().to(device)

    loss_function = nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(unet.parameters(), lr=1e-3)

    size = len(train_loader.dataset)  # type: ignore
    best_val_loss = 0

    for epoch in range(epochs):
        unet.train()
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            pred = unet(X)
            loss = loss_function(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch % 100 == 0:
                current = (batch + 1) * len(X)
                print(
                    f"epoch {epoch} loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]"
                )

        val_loss = evaluate(unet, val_loader, loss_function)
        print(f"epoch {epoch} validation loss: {val_loss:>7f}")

        # keep the best-performing checkpoint, not just the last one
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(unet.state_dict(), output_file)

    unet.load_state_dict(torch.load(output_file))
    test_loss = evaluate(unet, test_loader, loss_function)
    print(f"final test loss: {test_loss:>7f}")
