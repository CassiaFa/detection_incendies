import torch
import cv2

class cam():

    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        
    
    def load_model(self, model_name):

        '''
        Loads Yolo5 model from pytorch hub. If model_name is given loads a custom model, else loads the pretrained model.
        :return: Trained Pytorch model.
        '''
        
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name)  # local model
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        return model

    def image_detection(self, image):
        results = self.model(image)

        image = results.imgs[0]

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        n = len(labels)
        x_shape, y_shape = image.shape[1], image.shape[0]
        
        print("\n ========== \n", x_shape, y_shape, "\n ========== \n")

        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(image, f"{self.classes[int((labels[i]))]}  {row[4]:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return image

    def __call__(self, capture_index):

        self.cam = cv2.VideoCapture(capture_index)

        while True:
            _, frame = self.cam.read()
            
            frame = cv2.resize(frame, (640,640))

            results = self.model(frame)

            labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

            n = len(labels)
            x_shape, y_shape = frame.shape[1], frame.shape[0]
            for i in range(n):
                row = cord[i]
                if row[4] >= 0.3:
                    x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                    bgr = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                    cv2.putText(frame, f"{self.classes[int((labels[i]))]}  {row[4]:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
