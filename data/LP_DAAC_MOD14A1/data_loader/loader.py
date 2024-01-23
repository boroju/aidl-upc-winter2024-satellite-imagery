import os
import requests

def download_image(url, destination_folder):
    # Extract file name from URL
    file_name = url.split("/")[-1]
    file_path = os.path.join(destination_folder, file_name)

    # Download the image
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Image downloaded: {file_path}")
    else:
        print(f"Failed to download image from {url}")


def download_images_from_txt(txt_file, destination_folder):
    # Create the folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Read links from the text file
    with open(txt_file, 'r') as file:
        links = file.readlines()

    # Download each image
    for link in links:
        link = link.strip()  # Remove whitespace around the link
        download_image(link, destination_folder)


if __name__ == "__main__":
    # Get the project directory path (go back two levels from the current script location)
    project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

    # Change to the project directory
    os.chdir(project_directory)

    # Display the current directory after the change
    current_directory = os.getcwd()
    print(f"Current Directory: {current_directory}")

    data_path = "data/LP_DAAC_MOD14A1/"
    catalunya = "Spain/Catalunya/"
    sicily = "Italy/Sicily/"
    cordoba = "Argentina/Cordoba/"
    formosa = "Argentina/Formosa/"

    txt_cordoba = os.path.join(current_directory, data_path, cordoba,
                                 "cordoba-8974534166-download-browse-imagery.txt")

    destination_folder = "satellite_images/cordoba"

    download_images_from_txt(txt_cordoba, destination_folder)
