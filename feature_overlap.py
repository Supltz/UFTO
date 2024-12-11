import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from captum.attr import LayerGradCam

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import *
from data_prep import RAF_spliting
from expression_datasets import RAFDataset

# Load pre-trained weights into the models
def load_model_weights(model, path):
    model_dict = model.state_dict()
    pretrained = torch.load(path)
    if 'Trade_Off' in path:
        pretrained = {k.replace('base_model.', ''): v for k, v in pretrained.items()}
    pretrained = {k.replace('module.', ''): v for k, v in pretrained.items()}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)

def threshold_gradcam(gradcam, threshold=0.3):
    flat_gradcam = gradcam.flatten()
    top_70_threshold = np.percentile(flat_gradcam, 100 * (1 - threshold))
    # Only keep the top 70% features, others will be zero
    masked_gradcam = np.where(gradcam >= top_70_threshold, gradcam, 0)
    return masked_gradcam

# Calculate IoU for GradCAM
def calculate_iou(gradcam_1, gradcam_2, threshold=0.3):
    # Apply threshold to select the top 70% of features
    gradcam_1_flat = gradcam_1.flatten()
    gradcam_2_flat = gradcam_2.flatten()

    top_70_1 = np.percentile(gradcam_1_flat, 100 * (1 - threshold))
    top_70_2 = np.percentile(gradcam_2_flat, 100 * (1 - threshold))

    binary_1 = gradcam_1 >= top_70_1
    binary_2 = gradcam_2 >= top_70_2

    intersection = np.logical_and(binary_1, binary_2).sum()
    union = np.logical_or(binary_1, binary_2).sum()
    return intersection / union if union != 0 else 0

# Apply GradCAM and extract the most important 70% features
def apply_gradcam(model, layer, inputs, target_label):
    gradcam = LayerGradCam(model, layer)
    attributions = gradcam.attribute(inputs, target=target_label)
    return attributions[0].cpu().detach().numpy()

def find_extreme_indices(arr):
        # Find indices of the smallest, second smallest, largest, and second largest values
        min_index = np.argmin(arr)
        
        # Mask out the smallest and largest to find the second smallest and largest
        second_min_index = np.argmin(np.where(arr == arr[min_index], np.inf, arr))
        
        return min_index, second_min_index


def resize_cam(gradcam, target_size):
    gradcam = torch.tensor(gradcam).unsqueeze(0)
    gradcam_resized = F.interpolate(gradcam, size=(target_size[0], target_size[1]), mode='bilinear', align_corners=False)
    return gradcam_resized.squeeze().numpy() 

# Visualize and save the GradCAM images
def visualize_and_save(image, gradcam_emotion_list, gradcam_age, gradcam_gender, gradcam_race, iou_emotion_age_list, iou_emotion_gender_list, iou_emotion_race_list, index):
    # Create a new image with 3 channels representing each GradCAM result
    img = image.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())  # Normalize for visualization
    
    target_size = (img.shape[0], img.shape[1])  # Height and Width of the original image
    gradcam_age_resized = resize_cam(gradcam_age, target_size)
    gradcam_gender_resized = resize_cam(gradcam_gender, target_size)
    gradcam_race_resized = resize_cam(gradcam_race, target_size)
    gradcam_age_mask = threshold_gradcam(gradcam_age_resized)
    gradcam_gender_mask = threshold_gradcam(gradcam_gender_resized)
    gradcam_race_mask = threshold_gradcam(gradcam_race_resized)
 # Create and threshold emotion GradCAM masks
    gradcam_emotion_mask_list = [threshold_gradcam(resize_cam(cam, target_size)) for cam in gradcam_emotion_list]

 # Define the overlay function
    def overlay_gradcam_on_image(image, gradcam_mask, color, alpha=0.5):
        gradcam_mask = gradcam_mask / gradcam_mask.max()  # Normalize mask intensity
        heatmap = np.zeros_like(image)
        heatmap[:, :, 0] = gradcam_mask * color[0]
        heatmap[:, :, 1] = gradcam_mask * color[1]
        heatmap[:, :, 2] = gradcam_mask * color[2]
        return np.clip(np.where(gradcam_mask[..., None] > 0, (1 - alpha) * image + alpha * heatmap, image), 0, 1)
# Set up subplots dynamically: rows = number of gradcam_emotion_masks, columns = 3 for age, gender, race overlays
    fig, axes = plt.subplots(len(gradcam_emotion_mask_list), 3, figsize=(15, 5 * len(gradcam_emotion_mask_list)))
    # Ensure axes is always a 2D array even if there's only one row
    if len(gradcam_emotion_mask_list) == 1:
        axes = [axes]  # Wrap in a list to iterate consistently

# Iterate over each emotion GradCAM mask
    # Iterate over each emotion GradCAM mask

    for i, gradcam_emotion_mask in enumerate(gradcam_emotion_mask_list):
        # Overlay emotion GradCAM with age, gender, and race GradCAMs
        blended_emotion_age = overlay_gradcam_on_image(img, gradcam_emotion_mask, color=[1, 0, 0], alpha=0.6)
        blended_emotion_age = overlay_gradcam_on_image(blended_emotion_age, gradcam_age_mask, color=[0, 1, 0], alpha=0.6)

        blended_emotion_gender = overlay_gradcam_on_image(img, gradcam_emotion_mask, color=[1, 0, 0], alpha=0.6)
        blended_emotion_gender = overlay_gradcam_on_image(blended_emotion_gender, gradcam_gender_mask, color=[0, 1, 0], alpha=0.6)

        blended_emotion_race = overlay_gradcam_on_image(img, gradcam_emotion_mask, color=[1, 0, 0], alpha=0.6)
        blended_emotion_race = overlay_gradcam_on_image(blended_emotion_race, gradcam_race_mask, color=[0, 1, 0], alpha=0.6)

        # Display each overlay in its respective subplot with corresponding IoU score
        axes[i][0].imshow(blended_emotion_age)
        axes[i][0].set_title(f'Emotion-{i} vs Age IoU: {iou_emotion_age_list[i]:.2f}')
        axes[i][0].axis('off')

        axes[i][1].imshow(blended_emotion_gender)
        axes[i][1].set_title(f'Emotion-{i} vs Gender IoU: {iou_emotion_gender_list[i]:.2f}')
        axes[i][1].axis('off')

        axes[i][2].imshow(blended_emotion_race)
        axes[i][2].set_title(f'Emotion-{i} vs Race IoU: {iou_emotion_race_list[i]:.2f}')
        axes[i][2].axis('off')
    
   
    fig.savefig(f"./Feature_Entangle/correctly_classified/GradCAM/image_{index}.png")
    plt.close(fig)

def main():
    os.makedirs(f'./Feature_Entangle/correctly_classified/GradCAM', exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_path = '/vol/lian/datasets/basic/Image/aligned/'
    csv_path = '/vol/lian/datasets/Annotations/RAF-DB/annotations/'

    train_list, _, _ = RAF_spliting(data_path, csv_path, 'gender')

    pth_files = [f for f in os.listdir("./Trade_Off/checkpoints/") if f.endswith('.pth') and f.startswith('RAF')][:80]

    # Load the models
    emotion_model_list = []
    for i in range(len(pth_files)):
        emotion_model = Swin(pretrained=False, num_classes=7)
        emotion_model_list.append(emotion_model)
    
    gender_model = Swin(pretrained=False, num_classes=2)
    race_model = Swin(pretrained=False, num_classes=3)
    age_model = Swin(pretrained=False, num_classes=5)
    load_model_weights(gender_model, f'./Feature_Entangle/checkpoints/best_model_RAF_gender.pth')
    load_model_weights(race_model, f'./Feature_Entangle/checkpoints/best_model_RAF_race.pth')
    load_model_weights(age_model, f'./Feature_Entangle/checkpoints/best_model_RAF_age.pth')
    gender_model.to(device).eval()
    race_model.to(device).eval()
    age_model.to(device).eval()

    for i in tqdm(range(len(pth_files))):
        load_model_weights(emotion_model_list[i], './Trade_Off/checkpoints/' + pth_files[i])
        emotion_model_list[i].to(device).eval()


    # Load the validation data
    trainset = RAFDataset(train_list, data_path, mode='val')
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)

    correct_samples = {'images': [], 'labels': []}

    # Collect samples correctly classified by all models
    for inputs, labels in tqdm(trainloader, desc='Collecting correctly classified samples'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        correct_samples['images'].append(inputs.cpu().numpy())
        correct_samples['labels'].append(labels.cpu().numpy())

    correct_samples['images'] = np.concatenate(correct_samples['images'], axis=0)
    correct_samples['labels'] = np.concatenate(correct_samples['labels'], axis=0)
    
    correct_images = torch.tensor(correct_samples['images'], dtype=torch.float).to(device)
    correct_labels = torch.tensor(correct_samples['labels'], dtype=torch.long).to(device)
    
    print(f'Number of correctly classified images: {len(correct_images)}')

    # Apply GradCAM for all correctly classified images
    for i, (image, label) in enumerate(tqdm(zip(correct_images, correct_labels))):
        image = image.unsqueeze(0)  # Add batch dimension

        gradcam_age = apply_gradcam(age_model, age_model.layer4, image, target_label=label[2].item())
        gradcam_gender = apply_gradcam(gender_model, gender_model.layer4, image, target_label=label[1].item())
        gradcam_race = apply_gradcam(race_model, race_model.layer4, image, target_label=label[3].item())
        gradcam_emotion_list = []
        iou_emotion_age_list = []
        iou_emotion_gender_list = []
        iou_emotion_race_list = []
        all_list = []

        # Apply GradCAM for each model, passing the corresponding label
        for j in range(len(emotion_model_list)):
            gradcam_emotion = apply_gradcam(emotion_model_list[j], emotion_model_list[j].layer4, image, target_label=label[0].item())
            gradcam_emotion_list.append(gradcam_emotion)
            # Calculate IoU between different models (top 70% features)
            iou_emotion_age = calculate_iou(gradcam_emotion, gradcam_age)
            iou_emotion_gender = calculate_iou(gradcam_emotion, gradcam_gender)
            iou_emotion_race = calculate_iou(gradcam_emotion, gradcam_race)
            iou_emotion_age_list.append(iou_emotion_age)
            iou_emotion_gender_list.append(iou_emotion_gender)
            iou_emotion_race_list.append(iou_emotion_race)
            all_list.append([iou_emotion_age_list[j], iou_emotion_gender_list[j], iou_emotion_race_list[j]])

        all_list = np.array(all_list)

        if (

            np.any(np.all(all_list > 0.5, axis=1))
        ):
            visualize_and_save(image[0], gradcam_emotion_list, gradcam_age, gradcam_gender, gradcam_race,
                            iou_emotion_age_list, iou_emotion_gender_list, iou_emotion_race_list, i)

if __name__ == "__main__":
    main()