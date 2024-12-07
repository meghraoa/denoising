from data_loader import get_dataloader
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse
import sys
from pathlib import Path
from models import BaselineModel

ROOT_DIR = Path(__file__).parent
OUTPUT_FOLDER_NAME = "__output"
DATA_PATH = "data/source_separation"

def _get_parser(parser = None):
    if parser is None:
        parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("-e", "--exp", nargs="+", type=int, required=True, help="Experiment id")
    parser.add_argument("-o", "--output-dir", type=str, default=ROOT_DIR/OUTPUT_FOLDER_NAME, help="Output directory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    return parser

def train(
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        device="cpu",
        n_epochs=20,
        save_path="model.pth",
        log_dir="runs/experiment"
        ):
    """
    Fonction d'entraînement d'un modèle PyTorch pour la séparation de sources.

    Args:
        model (torch.nn.Module): Le modèle à entraîner.
        train_dataset (torch.utils.data.Dataset): Dataset pour l'entraînement.
        val_dataset (torch.utils.data.Dataset): Dataset pour la validation.
        criterion (torch.nn.Module): Fonction de perte.
        optimizer (torch.optim.Optimizer): Optimiseur.
        device (torch.device): Dispositif (CPU ou GPU).
        num_epochs (int): Nombre d'époques d'entraînement.
        batch_size (int): Taille des lots.
        save_path (str): Chemin pour enregistrer le modèle.

    Returns:
        dict: Historique des pertes pour l'entraînement et la validation.
    """
    writer = SummaryWriter(log_dir=log_dir)
    model.to(device)

    for epoch in range(n_epochs):
        pbar = tqdm(train_loader, unit="batches")
        model.train()
        train_loss = 0.0
        for batch_mix, batch_signal, batch_noise in pbar:
            batch_mix, batch_signal, batch_noise = batch_mix.to(device), batch_signal.to(device), batch_noise.to(device)
            batch_signal_pred, batch_noise_pred = model(batch_mix)
            loss = _compute_loss(criterion, batch_signal_pred, batch_signal, batch_noise_pred, batch_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        writer.add_scalar("Loss/Train", train_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_mix, batch_signal, batch_noise in valid_loader:
                batch_mix, batch_signal, batch_noise = batch_mix.to(device), batch_signal.to(device), batch_noise.to(device)
                batch_signal_pred, batch_noise_pred = model(batch_mix)
                loss = _compute_loss(criterion, batch_signal_pred, batch_signal, batch_noise_pred, batch_noise)
                val_loss += loss.item()

        val_loss /= len(valid_loader)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        print(f"Epoch {epoch+1}/{n_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        torch.save(model.state_dict(), save_path)

def _compute_loss(criterion, signal_pred, signal, noise_pred, noise):
    return (criterion(signal_pred, signal) + criterion(noise_pred, noise)) / 2

class DumbModel(nn.Module):
    def __init__(self):
        super(DumbModel, self).__init__()
        self.fc1 = nn.Linear(80000, 80000)
        self.fc2 = nn.Linear(80000, 80000)

    def forward(self, x):
        signal_pred = self.fc1(x)
        noise_pred = self.fc2(x)
        return signal_pred, noise_pred

# TEST
def main(argv):
    parser = _get_parser()
    args = parser.parse_args(argv)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Training dumb experiment on device {device}...")

    data_path = f'{DATA_PATH}/train_small'
    save_path = "models/model.pth"
    log_dir = "runs/dumb_experiment"

    # Chargement des données
    train_loader = get_dataloader(data_path, batch_size=8, shuffle=True)

    # Initialisation du modèle, de la fonction de perte et de l'optimiseur
    model = BaselineModel()
    criterion = nn.MSELoss()  # Perte MSE
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entraînement
    train(model, train_loader, train_loader, criterion, optimizer, device, n_epochs=2, save_path=save_path, log_dir=log_dir)

if __name__ == "__main__":
    main(sys.argv[1:])
