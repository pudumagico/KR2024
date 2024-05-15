import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torchvision.transforms.functional as F

from object_detection.object_detector import BaseObjectDetector


class OWLViTObjectDetector(BaseObjectDetector):
    def __init__(self, gpu, model="google/owlvit-large-patch14"):
        super().__init__(gpu)

        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model).to(gpu)
        self.processor = AutoProcessor.from_pretrained(model)

    def __should_merge__(self, box1, box2, overlap_threshold):
        YA1, XA1, YA2, XA2 = box1 
        YB1, XB1, YB2, XB2 = box2
        box1_area = (YA2 - YA1) * (XA2 - XA1)
        box2_area = (YB2 - YB1) * (XB2 - XB1)
        intersection_area = max(0, min(XA2, XB2) - max(XA1, XB1)) * max(0, min(YA2, YB2) - max(YA1, YB1))
        union_area = box1_area + box2_area - intersection_area

        if intersection_area / union_area > overlap_threshold:
            return True, (
                min(box1[0], box2[0]),
                min(box1[1], box2[1]),
                max(box1[2], box2[2]),
                max(box1[3], box2[3]),
            )

        return False, None

    def __merge_objects__(self, objects, overlap_threshold):
        for k in range(len(objects)):
            object1 = objects[k]
            for l in range(k+1, len(objects)):
                object2 = objects[l]
                if object1["name"] == object2["name"]:
                    is_merge, new_box = self.__should_merge__(
                        (object1['y'], object1['x'], object1['y']+object1['h'], object1['x']+object1['w']), 
                        (object2['y'], object2['x'], object2['y']+object2['h'], object2['x']+object2['w']), overlap_threshold
                    )
                    if is_merge:
                        objects[k] = None
                        objects[l] = {
                            "y": new_box[0],
                            "x": new_box[1],
                            "h": new_box[2]-new_box[0],
                            "w": new_box[3]-new_box[1],
                            "name": object1["name"],
                            "score": max(object1["score"], object2["score"])
                        }
                        break

        return [b for b in objects if b]

    def __choose_top_k_objects__(self, objects, k):
        return sorted(objects, key=lambda o: o["score"], reverse=True)[:k]

    @torch.no_grad()
    def detect_objects(self, image, classes, threshold=0.1, k=20):
        text_queries = classes
        inputs = self.processor(text=text_queries, images=F.to_pil_image(image), return_tensors="pt").to(self.gpu)

        outputs = self.model(**inputs)
        outputs.logits = outputs.logits.to("cpu")
        outputs.pred_boxes = outputs.pred_boxes.to("cpu")

        target_sizes = torch.tensor([(image.shape[1], image.shape[2])])
        results = self.processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
    
        scores = results["scores"].tolist()
        labels = results["labels"].tolist()
        boxes = results["boxes"].tolist()

        detected_objects = []
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            xmin, ymin, xmax, ymax = box
            detected_objects.append({
                "x": xmin,
                "y": ymin,
                "w": xmax - xmin,
                "h": ymax - ymin,
                "score": score,
                "name": text_queries[label]
            })

        merged_objects = self.__merge_objects__(detected_objects, overlap_threshold=0.6)

        if k is not None:
            top_k_objects = self.__choose_top_k_objects__(merged_objects, k)
        else:
            top_k_objects = merged_objects

        del inputs, outputs, results

        return top_k_objects