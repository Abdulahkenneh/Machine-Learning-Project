import numpy as np
def train_gan(generator, discriminator, gan, epochs, batch_size, low_res_images, high_res_images):
    for epoch in range(epochs):
        # Select a random batch of images
        idx = np.random.randint(0, low_res_images.shape[0], batch_size)
        low_res_imgs = low_res_images[idx]
        high_res_imgs = high_res_images[idx]

        # Generate high-resolution images
        generated_imgs = generator.predict(low_res_imgs)

        # Train discriminator
        d_loss_real = discriminator.train_on_batch(high_res_imgs, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_imgs, np.zeros((batch_size, 1)))

        # Train generator
        g_loss = gan.train_on_batch(low_res_imgs, np.ones((batch_size, 1)))

        print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss_real[0]:.4f}, G Loss: {g_loss:.4f}")

