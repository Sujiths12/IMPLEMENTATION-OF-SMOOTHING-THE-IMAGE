import cv2

# File path to the image
file_path = 'C:\\Users\\admin\\Downloads\\WhatsApp Image 2024-04-09 at 22.50.15_4d236316.jpg'

# Load an image
image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Error: Could not load image from {file_path}")
else:
    # Define Gaussian kernel size and standard deviation
    kernel_size = (5, 5)
    sigma = 1.5

    # Create 1D Gaussian kernel
    kernel_1d = cv2.getGaussianKernel(kernel_size[0], sigma)

    # Convert 1D kernel to 2D by taking the outer product
    kernel_2d = kernel_1d @ kernel_1d.T

    # Normalize the kernel
    kernel_2d /= kernel_2d.sum()

    # Pad the image to handle border effects
    padded_image = cv2.copyMakeBorder(image, kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2, cv2.BORDER_CONSTANT)

    # Convolve the image with the Gaussian kernel
    smoothed_image = cv2.filter2D(padded_image, -1, kernel_2d)

    # Display the original image and the smoothed image
    cv2.imshow('Original Image', image)
    cv2.imshow('Smoothed Image (Gaussian)', smoothed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
