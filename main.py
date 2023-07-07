import os
import glob
import matplotlib.pyplot as plt
import open_clip
import torch
from PIL import Image
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def load_open_clip(model_name: str = "ViT-B-32-quickgelu", pretrained: str = "laion400m_e32",
                   cache_dir: str = "biomed-clip-share",
                   device="cpu"):
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name, pretrained=pretrained,
                                                                                    cache_dir=cache_dir)
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess_val, tokenizer


# model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
#     'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
# tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
#
# # Downloading sample images
# snapshot_download("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", local_dir="biomed-clip-share")

# Example: Zero-shot classification
dataset_path = 'biomed-clip-share/example_data/biomed_image_classification_example_data'
template = 'this is a photo of '
labels = [
    'adenocarcinoma histopathology',
    'brain MRI',
    'covid line chart',
    'squamous cell carcinoma histopathology',
    'immunohistochemistry histopathology',
    'bone X-ray',
    'chest X-ray',
    'pie chart',
    'hematoxylin and eosin histopathology'
]

test_imgs = glob.glob(dataset_path + '/*')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model, preprocess_val, tokenizer = load_open_clip()
model.to(device)
model.eval()

context_length = 77

images = torch.stack([preprocess_val(Image.open(img)) for img in test_imgs]).to(device)
texts = tokenizer([template + l for l in labels], context_length=context_length).to(device)
with torch.no_grad():
    image_features, text_features, logit_scale = model(images, texts)

    logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)

    logits = logits.cpu().numpy()
    sorted_indices = sorted_indices.cpu().numpy()

top_k = -1

for i, img in enumerate(test_imgs):
    pred = labels[sorted_indices[i][0]]

    top_k = len(labels) if top_k == -1 else top_k
    print(img.split('/')[-1] + ':')
    for j in range(top_k):
        jth_index = sorted_indices[i][j]
        print(f'{labels[jth_index]}: {logits[i][jth_index]}')
    print('\n')


# Expected Output
def plot_images_with_metadata(images, metadata):
    num_images = len(images)
    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(10, 5 * num_images))

    for i, (img_path, metadata) in enumerate(zip(images, metadata)):
        img = Image.open(img_path)
        ax1 = axes[i, 0]
        ax1.imshow(img)
        ax1.axis('off')

        ax2 = axes[i, 1]
        ax2.axis('off')
        ax2.text(0, 0.5, f"{metadata['filename']}\n{metadata['top_probs']}", fontsize=10)

    plt.tight_layout()
    plt.show()


metadata_list = []

top_k = 3
for i, img in enumerate(test_imgs):
    pred = labels[sorted_indices[i][0]]
    img_name = img.split('/')[-1]

    top_probs = []
    top_k = len(labels) if top_k == -1 else top_k
    for j in range(top_k):
        jth_index = sorted_indices[i][j]
        top_probs.append(f"{labels[jth_index]}: {logits[i][jth_index] * 100:.1f}")

    metadata = {'filename': img_name, 'top_probs': '\n'.join(top_probs)}
    metadata_list.append(metadata)

plot_images_with_metadata(test_imgs, metadata_list)
