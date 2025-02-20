import cv2
from ultralytics import YOLO
import numpy as np
from sort import Sort
import torch


class DroneDetector:
    def __init__(self, device, path):
        self.device = device
        self.path = path
        self.model = self.load_model()
        self.sort = Sort(max_age=100, min_hits=4, iou_threshold=0.3)

    def load_model(self):
        model = YOLO("models/best.pt")
        model.to(self.device)
        model.fuse()
        return model

    def results(self, frame, model):
        results = model.predict(frame, conf=0.4)
        return results

    def get_results(self, results):
        arr = []
        for result in results[0]:
            bboxes = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            t_arr = [bboxes[0][0], bboxes[0][1], bboxes[0][2], bboxes[0][3], conf[0], classes[0]]
            arr.append(t_arr)
        return np.array(arr)

    def draw(self, frame, bboxes, idc, classes):
        for box, idx, cls in zip(bboxes, idc, classes):
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            cv2.putText(frame, "drone", (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.results(frame, self.model)
            array = self.get_results(results)

            if len(array) == 0:
                array = np.empty((0, 5))

            res = self.sort.update(array)

            bboxes = res[:, :-1]
            idc = res[:, -1].astype(int)
            classes = array[:, -1].astype(int)

            frame = self.draw(frame, bboxes, idc, classes)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


device = "mps" if torch.backends.mps.is_available() else "cpu"
path = "drone.mp4"
detector = DroneDetector(device, path)
detector()




