import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.datasets import ImageFolder
from train_model import train_model
from utils import save_model

# this ensures that the current MacOS version is at least 12.3+
mps_available = torch.backends.mps.is_available()
# this ensures that the current PyTorch installation was built with MPS activated.
mps_built = torch.backends.mps.is_built()

if mps_available and mps_built:
    device = torch.device("mps")

ImageFile.LOAD_TRUNCATED_IMAGES = True


def check_image(path):
    try:
        im = Image.open(path)
        return True
    except:
        return False


# train_data_path = "/.train/"

# Define some hyper-parameters
hparams = {
    'batch_size': 256,
    'num_epochs': 10,
    'test_batch_size': 256,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'log_interval': 10,
}

if __name__ == "__main__":
    # Get the directory path (go back three levels from the current script location)
    _dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    # Change directory
    os.chdir(_dir)
    # Get current directory after the change
    initial_path = os.getcwd()
    print(f"Initial Path: {initial_path}")

    train_path = initial_path + "/datasets/wildfire_prediction/train"
    test_path = initial_path + "/datasets/wildfire_prediction/test"
    valid_path = initial_path + "/datasets/wildfire_prediction/valid"

    # defining image transformations
    image_transforms = transforms.Compose([
        transforms.Resize((350, 350)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    train_data = ImageFolder(train_path, transform=image_transforms, is_valid_file=check_image)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=hparams['batch_size'], shuffle=True,
                                               drop_last=True)

    test_data = ImageFolder(test_path, transform=image_transforms, is_valid_file=check_image)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=hparams['batch_size'], shuffle=False,
                                              drop_last=True)

    eval_data = ImageFolder(valid_path, transform=image_transforms, is_valid_file=check_image)
    eval_loader = torch.utils.data.DataLoader(dataset=eval_data, batch_size=hparams['batch_size'], shuffle=False,
                                              drop_last=True)

    # Retrieve a sample from the dataset by simply indexing it
    img, label = train_data[0]
    print('Img shape: ', img.shape)
    print('Label: ', label)

    # Sample a BATCH from the dataloader by running over its iterator
    iter_ = iter(train_loader)
    bimg, blabel = next(iter_)
    print('Batch Img shape: ', bimg.shape)
    print('Batch Label shape: ', blabel.shape)
    print('Batch Img shape: ', bimg.shape)
    print('Batch Label shape: ', blabel.shape)
    print(f'The Batched tensors return a collection of {bimg.shape[0]} images \
    ({bimg.shape[1]} channel, {bimg.shape[2]} height pixels, {bimg.shape[3]} width \
    pixels)')
    print(f'In the case of the labels, we obtain {blabel.shape[0]} batched integers, one per image')

    print("Train:")
    print(f"Found {len(train_data)} images belonging to {train_data.classes} classes.")
    print("Test:")
    print(f"Found {len(test_data)} images belonging to {test_data.classes} classes.")
    print("Val:")
    print(f"Found {len(eval_data)} images belonging to {eval_data.classes} classes.")

    my_model = train_model(train_loader, test_loader, eval_loader, hparams).to(device)

    # Save the model checkpoint
    save_dir = "/checkpoints/my_model.pt"
    print(f"Saving model to {save_dir}...")
    save_model(my_model, save_dir)
    print("Model saved successfully!")
