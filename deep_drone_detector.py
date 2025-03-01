import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import torch
import random



class DeepDetecor:
    def __init__(self, path, device, color):
        self.path = path
        self.device = device
        self.model = self.load_model()
        self.names = self.model.names
        self.tracker = DeepSort(max_iou_distance=0.4, max_age=30)
        self.color = color

    def load_model(self):
        model = YOLO("models/best.pt")
        model.to(self.device)
        model.fuse()
        return model

    def results(self, frame):
        return self.model(frame)[0]

    def get_results(self, frame, results):
        res_array = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > 0.4:
                res_array.append(([int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1)], float(score), int(class_id)))


            tracks = self.tracker.update_tracks(raw_detections=res_array, frame=frame)

            for track in tracks:
                bboxes = track.to_ltrb()
                x1, y1, x2, y2 = bboxes
                idx = track.track_id
                class_id = track.get_det_class()

                text = f"{idx}{self.names[int(class_id)]}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), self.color, 2)
                cv2.putText(frame, text, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 2, self.color, 2)


                return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.results(frame)
            upd_frame = self.get_results(frame, results)
            cv2.imshow('Deep Detection', upd_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

path = "drone.mp4"
device = "mps" if torch.backends.mps.is_available() else "cpu"

color = []
for i in range(3):
    num = random.randint(0, 255)
    color.append(num)

color = tuple(color)
# print(color)


tracker = DeepDetecor(path, device, color)
tracker()


