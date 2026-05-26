from ultralytics import YOLO
import cv2

# 1. Load your trained model
# Check your 'runs' folder to see if it's 'train', 'train2', or 'train3'
model_path = 'runs/detect/train3/weights/best.pt' 
model = YOLO(model_path)

# 2. Run prediction on an image
# Replace 'test_image.jpg' with the path to a photo you want to test
results = model.predict(source='test_image.jpg', conf=0.5, save=True)

# 3. Show the result
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    cv2.imshow('Traffic Light Detection', im_array)
    cv2.waitKey(0) # Press any key to close the window

cv2.destroyAllWindows()