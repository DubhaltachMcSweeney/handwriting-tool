import cv2
import os
import matplotlib.pyplot as plt

def load_image(path):
    """ This reads an image from a path using the library cv2"""
    if not os.path.exists(path):
        print("File not found.")
        return None
    
    image = cv2.imread(path)
    
    if image is None:
        print("Error loading image.")
        return None
    
    return image

def main():
    """ This runs the image loader, it loads an image with cv2 and then displays with matplotlib"""
    path = input("Enter path to handwritten image: ")

    image = load_image(path)
    
    if image is not None:
        print("Image loaded successfully!")
        plt.imshow(image, cmap="grey")
        plt.show()

if __name__ == "__main__":
    main()