import os
import random
import yaml

# Define paths and parameters
data_dir = 'data/imagenet-a'
test_size = 1500

# Function to get all images and corresponding targets
def get_images_and_targets(data_dir):
    images = []
    targets = []
    class_names = os.listdir(data_dir)
    class_names.sort()
    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                if image_name.endswith('.jpg'):
                    images.append(os.path.join(data_dir, class_name, image_name))
                    targets.append(class_index)
    return images, targets

# Get all images and targets
images, targets = get_images_and_targets(data_dir)



# Shuffle the images and targets together
combined = list(zip(images, targets))
random.shuffle(combined)
images[:], targets[:] = zip(*combined)

# Randomly select test images
test_images = images[:test_size]
test_targets = targets[:test_size]

# Remaining images and targets for training
train_images = images[test_size:]
train_targets = targets[test_size:]

# Prepare data for yaml files
test_data = {
    'data': test_images,
    'targets': test_targets
}

train_data = {
    'data': train_images,
    'targets': train_targets
}

# Save test.yaml
with open('dataloaders/splits/imagenet-a_test.yaml', 'w') as test_file:
    yaml.dump(test_data, test_file)

# Save train.yaml
with open('dataloaders/splits/imagenet-a_train.yaml', 'w') as train_file:
    yaml.dump(train_data, train_file)

print('Generated test.yaml and train.yaml successfully.')