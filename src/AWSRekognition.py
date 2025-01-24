import boto3
import cv2


class AWSRekognition:
    def __init__(self):
        self.rekognition_client = boto3.client('rekognition', region_name='eu-central-1')
    def label_image(self, img):
        result = img.copy()
        print("Detect Labels")
        # Convert the image to bytes
        _, image_bytes = cv2.imencode('.jpg', img)

        # Call the detect_labels method
        response = self.rekognition_client.detect_labels(
            Image={
                'Bytes': image_bytes.tobytes()
            },
            MaxLabels=20,
            MinConfidence=90
        )

        # Print the labels
        for label in response['Labels']:
            print(label['Name'], label['Confidence'])

        # Draw bounding boxes around the detected objects
        for label in response['Labels']:
            for instance in label['Instances']:
                bounding_box = instance['BoundingBox']
                x = int(bounding_box['Left'] * result.shape[1])
                y = int(bounding_box['Top'] * result.shape[0])
                w = int(bounding_box['Width'] * result.shape[1])
                h = int(bounding_box['Height'] * result.shape[0])
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return result
