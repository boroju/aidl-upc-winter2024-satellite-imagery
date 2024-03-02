import os
import torch
from torchvision import transforms
from PIL import Image
from model import WildfireBinClassifier

device = "cpu"


def preprocessing_transforms(pil_image: Image.Image) -> torch.Tensor:
    # defining image transformations
    preprocess = transforms.Compose([
        transforms.Resize((350, 350)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    torch_image = preprocess(pil_image)
    try:
        assert isinstance(torch_image, torch.Tensor)
        assert list(torch_image.shape) == [3, 350, 350]
        print("preprocessing_transforms: OK")
    except Exception:
        raise Exception("Did you apply required transformations?")

    return torch_image


def predict_image(
        torch_image: torch.tensor,
        model: torch.nn.Module
):
    x = torch_image.to(device)  # move image to device
    x = x.unsqueeze(0)  # add batch dimension

    # Make a prediction
    output = model(x)
    output = torch.softmax(output, dim=1)  # Compute the softmax to get probabilities

    # labels = {'nowildfire': 0, 'wildfire': 1}
    class_names = ['nowildfire', 'wildfire']

    # Get the index of the class with the highest probability
    predicted_class_idx = torch.argmax(output).item()

    # Get the corresponding class name
    predicted_class_name = class_names[predicted_class_idx]

    print(f"Predicted class: {predicted_class_name}")


if __name__ == "__main__":
    print()
    print(f"========================================================================")
    print(f"== Wildfire Binary Classifier with Kaggle Wildfire Prediction Dataset ==")
    print(f"========================================================================")
    print(f"== PREDICTION PART ==")
    # Get current directory
    initial_path = os.getcwd()
    print(f"Initial Path: {initial_path}")
    print(f"Device: {device}")

    # Define the path to your saved model file
    model_path = './checkpoints/'
    model_name = 'my_model.pt'
    my_model = model_path + model_name

    # Load the model state dictionary
    state_dict = torch.load(my_model, map_location=torch.device(device))

    # Reconstruct the model architecture
    model = WildfireBinClassifier()
    model.load_state_dict(state_dict)

    # Ensure the model is in evaluation mode
    model.eval()

    # Load images for prediction
    prediction_folder = './prediction/'
    img1 = 'test_nw_-73.47513,45.58354.jpg'
    img2 = 'test_nw_-73.58979,45.483765.jpg'
    img3 = 'test_w_-62.56176,51.29047.jpg'
    img4 = 'test_w_-63.3175,51.3397.jpg'
    pil_image1 = Image.open(prediction_folder + img1)
    pil_image2 = Image.open(prediction_folder + img2)
    pil_image3 = Image.open(prediction_folder + img3)
    pil_image4 = Image.open(prediction_folder + img4)

    # Apply preprocessing transformations
    tensor_img1 = preprocessing_transforms(pil_image1)
    tensor_img2 = preprocessing_transforms(pil_image2)
    tensor_img3 = preprocessing_transforms(pil_image3)
    tensor_img4 = preprocessing_transforms(pil_image4)

    print()

    # Predictions
    print(f"Predictions:")
    print(f"============")
    print()
    print(f"Image 1: {img1}")
    print(f"Expected: nowildfire")
    predict_image(tensor_img1, model)
    print()
    print(f"Image 2: {img2}")
    print(f"Expected: nowildfire")
    predict_image(tensor_img2, model)
    print()
    print(f"Image 3: {img3}")
    print(f"Expected: wildfire")
    predict_image(tensor_img3, model)
    print()
    print(f"Image 4: {img4}")
    print(f"Expected: wildfire")
    predict_image(tensor_img4, model)
