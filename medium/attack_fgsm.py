import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the fashion classifier model
model = tf.keras.models.load_model('fashion_classifier.h5')

# Load class names
with open('class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

def load_and_preprocess_image(image_path):
    """Load and preprocess image for the model"""
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Fashion MNIST is 28x28
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    return img_array, img

def fgsm_attack(model, image, epsilon=0.01):
    """
    Fast Gradient Sign Method (FGSM) Attack
    
    Args:
        model: The target model
        image: Input image (preprocessed)
        epsilon: Perturbation magnitude
    """
    # Convert to tensor
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    
    # Get the original prediction
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(image_tensor)
        original_class = tf.argmax(prediction[0])
        
        # Calculate loss for the original predicted class
        loss = prediction[0][original_class]
    
    # Calculate gradient
    gradient = tape.gradient(loss, image_tensor)
    
    # Create adversarial example using FGSM
    signed_grad = tf.sign(gradient)
    adversarial_image = image_tensor - epsilon * signed_grad
    
    # Clip values to valid range [0, 1]
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
    
    return adversarial_image.numpy()

def iterative_fgsm_attack(model, image, epsilon=0.01, iterations=10, alpha=0.001):
    """
    Iterative FGSM (I-FGSM) - More powerful attack
    
    Args:
        model: The target model
        image: Input image
        epsilon: Maximum perturbation
        iterations: Number of iterations
        alpha: Step size per iteration
    """
    adversarial_image = tf.convert_to_tensor(image, dtype=tf.float32)
    original_image = tf.identity(adversarial_image)
    
    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adversarial_image)
            prediction = model(adversarial_image)
            predicted_class = tf.argmax(prediction[0])
            loss = prediction[0][predicted_class]
        
        gradient = tape.gradient(loss, adversarial_image)
        signed_grad = tf.sign(gradient)
        
        # Update adversarial image
        adversarial_image = adversarial_image - alpha * signed_grad
        
        # Project back to epsilon-ball around original image
        perturbation = adversarial_image - original_image
        perturbation = tf.clip_by_value(perturbation, -epsilon, epsilon)
        adversarial_image = original_image + perturbation
        
        # Clip to valid range
        adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
    
    return adversarial_image.numpy()

def targeted_fgsm_attack(model, image, target_class, epsilon=0.01, iterations=20):
    """
    Targeted FGSM - Force model to predict a specific class
    """
    adversarial_image = tf.convert_to_tensor(image, dtype=tf.float32)
    original_image = tf.identity(adversarial_image)
    
    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adversarial_image)
            prediction = model(adversarial_image)
            # Maximize the target class probability (minimize negative)
            loss = -prediction[0][target_class]
        
        gradient = tape.gradient(loss, adversarial_image)
        signed_grad = tf.sign(gradient)
        
        adversarial_image = adversarial_image - 0.001 * signed_grad
        
        perturbation = adversarial_image - original_image
        perturbation = tf.clip_by_value(perturbation, -epsilon, epsilon)
        adversarial_image = original_image + perturbation
        adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
    
    return adversarial_image.numpy()

def main():
    # Find the secret image
    images_dir = 'images'
    secret_image_path = None
    
    # Look for secret image (adjust pattern as needed)
    for filename in os.listdir(images_dir):
        if 'secret' in filename.lower() or filename.startswith('secret'):
            secret_image_path = os.path.join(images_dir, filename)
            break
    
    if not secret_image_path:
        # Try first image in directory
        files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            secret_image_path = os.path.join(images_dir, files[0])
    
    if not secret_image_path:
        print("❌ No image found!")
        return
    
    print(f"🔍 Loading image: {secret_image_path}")
    
    # Load and preprocess
    original_image, pil_img = load_and_preprocess_image(secret_image_path)
    
    # Get original prediction
    original_pred = model.predict(original_image, verbose=0)
    original_class = np.argmax(original_pred[0])
    original_confidence = original_pred[0][original_class]
    
    print(f"\n📊 Original Prediction:")
    print(f"   Class: {class_names[original_class]}")
    print(f"   Confidence: {original_confidence:.4f}")
    
    # Try different attack methods
    print("\n🎯 Launching FGSM Attack...")
    adversarial_fgsm = fgsm_attack(model, original_image, epsilon=0.05)
    
    print("\n🎯 Launching Iterative FGSM Attack...")
    adversarial_ifgsm = iterative_fgsm_attack(model, original_image, epsilon=0.05, iterations=20)
    
    # Test adversarial examples
    attacks = [
        ("FGSM", adversarial_fgsm),
        ("I-FGSM", adversarial_ifgsm)
    ]
    
    for attack_name, adv_image in attacks:
        adv_pred = model.predict(adv_image, verbose=0)
        adv_class = np.argmax(adv_pred[0])
        adv_confidence = adv_pred[0][adv_class]
        
        print(f"\n🔥 {attack_name} Results:")
        print(f"   New Class: {class_names[adv_class]}")
        print(f"   Confidence: {adv_confidence:.4f}")
        
        if adv_class != original_class:
            print(f"   ✅ SUCCESS! Model fooled!")
            print(f"   Changed from '{class_names[original_class]}' to '{class_names[adv_class]}'")
            
            # Calculate perturbation
            perturbation = np.abs(adv_image - original_image).mean()
            print(f"   Average perturbation: {perturbation:.6f}")
        else:
            print(f"   ⚠️  Same class - try increasing epsilon")
    
    # Try targeted attacks to different classes
    print("\n🎯 Trying Targeted Attacks...")
    for target_idx in range(min(3, len(class_names))):
        if target_idx != original_class:
            targeted_adv = targeted_fgsm_attack(model, original_image, target_idx, epsilon=0.1)
            targeted_pred = model.predict(targeted_adv, verbose=0)
            targeted_class = np.argmax(targeted_pred[0])
            
            if targeted_class == target_idx:
                print(f"   ✅ Successfully targeted '{class_names[target_idx]}'!")

if __name__ == "__main__":
    main()