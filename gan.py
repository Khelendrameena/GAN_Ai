import time
import torch
import torch.nn as nn
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from gensim.models import Word2Vec
from PIL import Image


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class UNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetGenerator, self).__init__()
        # Encoder (downsampling)
        self.down1 = self.conv_block(in_channels, 64, down=True)
        self.down2 = self.conv_block(64, 128, down=True)
        self.down3 = self.conv_block(128, 256, down=True)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder (upsampling)
        self.up3 = self.conv_block(512 + 256, 256, up=True)
        self.up2 = self.conv_block(256 + 128, 128, up=True)
        self.up1 = self.conv_block(128 + 64, 64, up=True)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch, down=False, up=False):
        layers = []
        if down:
            layers.append(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False))
        elif up:
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False))
        else:
            layers.append(nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False))

        layers.extend([
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ])
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        # Bottleneck
        bottleneck = self.bottleneck(d3)

        # Decoder
        u3 = self.up3(torch.cat([bottleneck, d3], dim=1))
        u2 = self.up2(torch.cat([u3, d2], dim=1))
        u1 = self.up1(torch.cat([u2, d1], dim=1))

        return self.final_conv(u1)

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            self.conv_block(in_channels * 2, 64, norm=False),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def conv_block(self, in_ch, out_ch, norm=True):
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))

# Initialize models
generator = UNetGenerator(3, 3)
discriminator = Discriminator(3)

# Loss functions
adversarial_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop (simplified)
def train(input_image, target_image, i):
    # Train Discriminator
    optimizer_D.zero_grad()

    real_output = discriminator(input_image, target_image)
    fake_image = generator(input_image)
    fake_output = discriminator(input_image, fake_image.detach())

    d_loss = (adversarial_loss(fake_output, torch.zeros_like(fake_output)) +
              adversarial_loss(real_output, torch.ones_like(real_output))) / 2

    d_loss.backward()
    optimizer_D.step()

    # Train Generator
    optimizer_G.zero_grad()

    fake_output = discriminator(input_image, fake_image)
    g_loss = adversarial_loss(fake_output, torch.ones_like(fake_output)) + \
             100 * l1_loss(fake_image, target_image)  # L1 loss weight

    g_loss.backward()
    optimizer_G.step()

    return i, d_loss.item(), g_loss.item()


def model_save():
    item_path = os.path.join('.', 'model')
    if not os.path.isdir(item_path):
          os.makedirs('model')
    final_generator_path = os.path.join('model/', 'generator.pth')
    final_discriminator_path = os.path.join('model/', 'discriminator.pth')
    torch.save(generator.state_dict(), final_generator_path)
    torch.save(discriminator.state_dict(), final_discriminator_path)
    print("Final model saved")

def extract_frames(video_path, output_dir,p):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the video
    cap = cv2.VideoCapture(video_path)

    frame_count = p

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Save frame as an image
        frame = cv2.resize(frame,(256,256))
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()

class Pix2PixDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.input_dir = os.path.join(root_dir, 'input')
        self.target_dir = os.path.join(root_dir, 'target')
        self.image_files = os.listdir(self.input_dir)

    def __len__(self):
        return len(self.image_files)

    # This method was defined outside of the class
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        input_path = os.path.join(self.input_dir, img_name)
        target_path = os.path.join(self.target_dir, img_name)

        input_image = Image.open(input_path).convert('RGB')
        target_image = Image.open(target_path).convert('RGB')

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image

def __getitem__(self, idx):
    img_name = self.image_files[idx]
    input_path = os.path.join(self.input_dir, img_name)
    target_path = os.path.join(self.target_dir, img_name)

    input_image = Image.open(input_path).convert('RGB')
    target_image = Image.open(target_path).convert('RGB')

    if self.transform:
        input_image = self.transform(input_image)
        target_image = self.transform(target_image)

    return input_image, target_image



def model_train(epochs):
    dataset = Pix2PixDataset(root_dir='data/', transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for epoch in range(epochs):
        for i, (input_image, target_image) in enumerate(dataloader):
            i, d_loss, g_loss = train(input_image, target_image, i)
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}], D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

def mot_gan(dir_path, x, y):
    # Get the list of files in the directory
    files = sorted(os.listdir(dir_path))

    # If x > 0, delete first x files
    if x > 0:
        files_to_delete = files[:x]  # First x files
    # If y > 0, delete last y files
    elif y > 0:
        files_to_delete = files[-y:]  # Last y files
    else:
        files_to_delete = []  # No files to delete

    # Loop through and delete the specified files
    for file in files_to_delete:
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            #print(f"Deleted: {file_path}")
        else:
            print(f"Skipped: {file_path} is not a file")

    #print(f"Deleted {len(files_to_delete)} files from {dir_path}")




def model_dataset(video_path,smo):
    item_path = os.path.join('.', 'data')
    if not os.path.isdir(item_path):
        os.makedirs('data')
    extract_frames(video_path,'data/input',-smo)
    extract_frames(video_path,'data/target',0)
    mot_gan('data/input',smo,0)
    mot_gan('data/target',0,smo)

def model_generate(input_image_path):
    # लोड प्रशिक्षित मॉडल
    generator.load_state_dict(torch.load('model/generator.pth'))
    generator.eval()  # मूल्यांकन मोड में सेट करें
    input_image = Image.open(input_image_path).convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0)  # बैच डाइमेंशन जोड़ें

    # इमेज जनरेट करें
    with torch.no_grad():
        output_tensor = generator(input_tensor)

    # आउटपुट टेंसर को इमेज में कनवर्ट करें
    output_image = transforms.ToPILImage()(output_tensor.squeeze(0) * 0.5 + 0.5)
    #os.makedirs('ganrated')
    unique_name = time.strftime("%Y%m%d%H%M%S")

    # आउटपुट इमेज सेव कर

    output_image.save(f'generated/{unique_name}.jpg')
    return f'generated/{unique_name}.jpg'


# Function to generate images using the GAN model
def model_generate(input_image_path):
    # Load the pre-trained model
    generator.load_state_dict(torch.load('model/generator.pth'))
    generator.eval()  # Set model to evaluation mode

    input_image = Image.open(input_image_path).convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension

    # Generate image
    with torch.no_grad():
        output_tensor = generator(input_tensor)

    # Convert output tensor to image
    output_image = transforms.ToPILImage()(output_tensor.squeeze(0) * 0.5 + 0.5)

    # Create a unique name using timestamp
    unique_name = time.strftime("%Y%m%d%H%M%S")

    # Save the output image
    output_path = f'generated/{unique_name}.jpg'
    output_image.save(output_path)

    return output_path

# Function to run the GAN model and generate multiple frames and create a video
def model_gan(path='te_im/main.jpg',frame_count=10):
    item_path = os.path.join('.', 'generated')
    if not os.path.isdir(item_path):
        os.makedirs('generated')
    for i in range(0, frame_count):
        path = model_generate(path)

    # Directory where images are stored
    image_folder = 'generated'

    # Directory to save the video
    output_folder = 'result'

    # Ensure the output directory exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Video file name and path
    video_name = os.path.join(output_folder, f'{time.strftime("%Y%m%d%H%M%S")}.mp4')

    # Get the list of images and sort them
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()  # Ensure the images are sorted in the right order

    # Create the video from the generated images
    if images:
        # Read the first image to get the frame size
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        # Initialize video writer
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

        # Add each image as a frame in the video
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        # Release the video writer
        video.release()

        print(f"Video saved successfully at: {video_name}")
    else:
        print("No images found in the directory.")
    os.system['rm -r generated']

# Function to convert sentence to image
def se_im(sentence, output_dir='te_im', image_size=(256, 256), vector_size=100):
    """
    Converts a sentence to a 256x256 RGB image using Word2Vec embeddings.

    Parameters:
    - sentence (str): Input sentence to convert to an image.
    - output_dir (str): Directory where the image will be saved.
    - image_size (tuple): Size of the output image (default is 256x256).
    - vector_size (int): Size of the Word2Vec vector for each word (default is 100).

    Returns:
    - None: The image is saved in the specified output directory.
    """
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Tokenize the sentence into words
    words = sentence.lower().split()

    # Train a simple Word2Vec model (or use a pre-trained model)
    model = Word2Vec([words], vector_size=vector_size, window=5, min_count=1, workers=1)

    # Get embeddings for each word in the sentence
    embeddings = []
    for word in words:
        if word in model.wv:
            embeddings.append(model.wv[word])
        else:
            embeddings.append(np.zeros(vector_size))  # If word not in vocabulary, use zero vector

    # Convert the list of embeddings into a numpy array
    embeddings = np.array(embeddings)

    # Combine embeddings by averaging them
    sentence_embedding = np.mean(embeddings, axis=0)

    # Repeat the sentence embedding to fit 256x256x3 (image_size[0] * image_size[1] * 3)
    required_size = image_size[0] * image_size[1] * 3
    embedding_size = sentence_embedding.size

    # If the embedding is smaller, repeat it to fill the required size
    repeated_embedding = np.tile(sentence_embedding, required_size // embedding_size + 1)[:required_size]

    # Reshape the repeated embedding into the specified image size
    rgb_image_array = repeated_embedding.reshape(image_size[0], image_size[1], 3)

    # Normalize the values to fit in the range [0, 255]
    rgb_image_array = (rgb_image_array - rgb_image_array.min()) / (rgb_image_array.max() - rgb_image_array.min()) * 255
    rgb_image_array = rgb_image_array.astype(np.uint8)

    # Convert the array to an image
    img = Image.fromarray(rgb_image_array, 'RGB')

    # Save the image
    img_filename = 'sentence_to_image.png'
    img.save(os.path.join(output_dir,'main.jpg'))

    #print(f"Image saved as '{img_filename}' in {output_dir} folder.")
