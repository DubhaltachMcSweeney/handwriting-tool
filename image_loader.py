import cv2
import os

def load_image(path):
    if not os.path.exists(path):
        print("File not found.")
        return None
    
    image = cv2.imread(path)
    
    if image is None:
        print("Error loading image.")
        return None
    
    return image

def main():
    path = input("Enter path to handwritten image: ")
    image = load_image(path)
    
    if image is not None:
        print("Image loaded successfully!")
        cv2.imshow("Input Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()