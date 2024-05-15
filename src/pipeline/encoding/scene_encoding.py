from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import crop, resize, pad
from pipeline.concept_extraction import extract_classes, extract_attributes, extract_relations
from pipeline.bounding_box_optimization import get_object_bboxes, get_pair_bboxes
from pipeline.utils import cleanup_whitespace, sanitize_asp
import math
import torch
import json
import itertools


with open('../data/metadata/gqa_all_attribute.json') as f:
    all_attributes = json.load(f)

with open('../data/metadata/gqa_all_class.json') as f:
    all_classes = json.load(f)
    all_child_classes = [c.replace("_", " ") for c in itertools.chain(*all_classes.values())]


def bboxes_to_image_crops(bboxes, image, model, mode="pad"):
    bbox_crops = []
    for bbox in bboxes:
        y, x, h, w = int(bbox[0]), int(bbox[1]), int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])
        bbox_crop = crop(image, y, x, h, w)

        if mode == "pad":
            # resize and scale (maintain aspect ratio)
            if h > w:
                resize_dimensions = (
                    model.img_size, 2*round((model.img_size*w/h)/2))
            else:
                resize_dimensions = (
                    2*round((model.img_size*h/w)/2), model.img_size)
            bbox_crop = resize(bbox_crop, resize_dimensions, antialias=True)

            # pad the image to square dimensions
            bbox_crop = pad(bbox_crop, ((
                model.img_size - resize_dimensions[1])//2, (model.img_size - resize_dimensions[0])//2))

        elif mode == "scale":
            # resize and scale the image to the target dimensions
            bbox_crop = resize(
                bbox_crop, (model.img_size, model.img_size), antialias=True)
        else:
            raise RuntimeError("Unsupported image processing mode!")

        bbox_crops.append(bbox_crop)

    return bbox_crops


def prob_to_asp_weight(prob):
    return int(min(-1000*math.log(prob), 5000))

def __should_merge__(box1, box2, overlap_threshold):
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

def merge_detected_objects(objects_a, objects_b):
    objects = [*objects_a]
    for ob in objects_b:
        alread_present = False
        for oa in objects_a:
            categories_a = {cat for cat in all_classes if oa["name"] in all_classes[cat]}
            categories_b = {cat for cat in all_classes if ob["name"] in all_classes[cat]}
            if len(categories_a & categories_b) > 0:
                is_merge, _ = __should_merge__(
                        (oa['y'], oa['x'], oa['y']+oa['h'], oa['x']+oa['w']), 
                        (ob['y'], ob['x'], ob['y']+ob['h'], ob['x']+ob['w']), 
                        overlap_threshold=0.7
                    )
                if is_merge:
                    alread_present = True
        if not alread_present:
            objects.append(ob)

    return objects


def get_article(name):
    return "an" if any(name.startswith(v) for v in ["a", "e", "i", "o", "u"]) else "a"


def detect_objects_question_driven(image, classes, object_detector):
    objects = []
    for clazz in classes["classes"]:
        detected_objects = object_detector.detect_objects(image, [clazz.replace("_", " ")], threshold=0.03, k=5)
        objects.extend(detected_objects)

    for category in classes["categories"]:
        detected_objects = object_detector.detect_objects(image, [c.replace("_", " ") for c in all_classes[category]], threshold=0.03, k=5)
        objects = merge_detected_objects(objects, detected_objects)

    if classes["all"]:
        detected_objects = object_detector.detect_objects(image, all_child_classes, threshold=0.03, k=25)
        objects = merge_detected_objects(objects, detected_objects)
    return objects

@torch.no_grad()
def encode_scene(question, model, object_detector):
    scene_encoding = ""
    
    attributes, standalone_values = extract_attributes(question)
    num_attr_values = sum(len(all_attributes.get(attr, [])) for attr in attributes)
    num_standalone_values = len(standalone_values)
    classes = extract_classes(question)
    relations = extract_relations(question)
    num_relations = len(relations)

    for attr in attributes:
        scene_encoding += f"is_attr({cleanup_whitespace(attr)}).\n"
        for val in all_attributes.get(attr, []):
            scene_encoding += f"is_attr_value({cleanup_whitespace(attr)}, {cleanup_whitespace(val)}).\n"
        scene_encoding += "\n"

    scene_encoding += "\n"

    image = read_image(f"../data/images/{question['imageId']}.jpg", ImageReadMode.RGB)
    image_size = {'w': image.shape[2], 'h': image.shape[1]}

    objects = detect_objects_question_driven(image, classes, object_detector)
    object_items = [(f"o{i}", o) for i, o in enumerate(objects)]
    num_objects = len(objects)

    if (len(attributes) > 0 or len(standalone_values) > 0) and len(objects) > 0:
        object_bboxes = get_object_bboxes(objects, image_size)
        obj_bbox_crops = bboxes_to_image_crops(object_bboxes, image, model)
       
        neutral_prompts = [f"a pixelated picture of {get_article(obj['name'])} {obj['name']}" for obj in objects]
        attr_prompts = [f"a pixelated picture of {get_article(val)} {val} {obj['name']}"
                        for obj in objects
                        for attr in attributes
                        for val in all_attributes.get(attr, [])]
        standalone_value_prompts = [f"a pixelated picture of {get_article(val)} {val} {obj['name']}"
                                    for obj in objects
                                    for val in standalone_values]
        
        obj_logits_per_image = model.score(obj_bbox_crops, [*neutral_prompts, *attr_prompts, *standalone_value_prompts])
            
    if len(relations) > 0 and len(objects) > 1:
        rel_bboxes, rel_bbox_indices = get_pair_bboxes(objects, merge_threshold=0.6)
        rel_bbox_crops = bboxes_to_image_crops(rel_bboxes, image, model)
    
    # add attributes derived from object detection (names, vposition/hposition)
    for o1, (oid1, object1) in enumerate(object_items):
        scene_encoding += f"object({oid1}).\n"
        scene_encoding += f"has_obj_weight({oid1}, {prob_to_asp_weight(object1['score'])}).\n"

        scene_encoding += f"has_attr({oid1}, class, {sanitize_asp(object1['name'])}).\n"
        for category in all_classes: 
            if sanitize_asp(object1["name"]) in all_classes[category]:
                scene_encoding += f"has_attr({oid1}, class, {sanitize_asp(category)}).\n"
        
        scene_encoding += f"has_attr({oid1}, name, {sanitize_asp(object1['name'])}).\n"

        if (object1['x'] + object1['w']/2) > image_size["w"]/3*2:
            scene_encoding += f"has_attr({oid1}, hposition, right).\n"
        elif (object1['x'] + object1['w']/2) > image_size["w"]/3:
            scene_encoding += f"has_attr({oid1}, hposition, middle).\n"
        else:
            scene_encoding += f"has_attr({oid1}, hposition, left).\n"

        if (object1['y'] + object1['h']/2) > image_size["h"]/3*2:
            scene_encoding += f"has_attr({oid1}, vposition, bottom).\n"
        elif (object1['y'] + object1['h']/2) > image_size["h"]/3:
            scene_encoding += f"has_attr({oid1}, vposition, middle).\n"
        else:
            scene_encoding += f"has_attr({oid1}, vposition, top).\n"
        scene_encoding += "\n"

        neutral_indices = num_objects
        if len(attributes) > 0:    
            attr_scores = torch.stack([
                obj_logits_per_image[o1, neutral_indices+o1*(num_attr_values):neutral_indices+(o1+1)*num_attr_values],
                obj_logits_per_image[o1, o1].expand(num_attr_values)
            ])
            attr_probs = torch.nn.functional.softmax(attr_scores, dim=0)  

            # print(f"{num_attr_values}, {attr_probs.shape}")

            j = 0
            for attr in attributes:
                for val in all_attributes.get(attr, []): 
                    scene_encoding += f"{{has_attr({oid1}, {cleanup_whitespace(attr)}, {cleanup_whitespace(val)})}}.\n"
                    scene_encoding += f":~ has_attr({oid1}, {cleanup_whitespace(attr)}, {cleanup_whitespace(val)}). [{prob_to_asp_weight(attr_probs[0,j])}, ({oid1}, {cleanup_whitespace(attr)}, {cleanup_whitespace(val)})]\n"
                    scene_encoding += f":~ not has_attr({oid1}, {cleanup_whitespace(attr)}, {cleanup_whitespace(val)}). [{prob_to_asp_weight(attr_probs[1,j])}, ({oid1}, {cleanup_whitespace(attr)}, {cleanup_whitespace(val)})]\n"
                    j += 1
                
            scene_encoding += "\n"

            del attr_scores, attr_probs

        if len(standalone_values) > 0:
            attr_indices = num_objects*num_attr_values
            standalone_scores = torch.stack([
                obj_logits_per_image[o1, (neutral_indices+attr_indices)+o1*num_standalone_values:(neutral_indices+attr_indices)+(o1+1)*num_standalone_values],
                obj_logits_per_image[o1, o1].expand(num_standalone_values)
            ])
            standalone_probs = torch.nn.functional.softmax(standalone_scores, dim=0)

            # print(f"{num_standalone_values}, {standalone_probs.shape}")

            k = 0
            for standalone_value_ in standalone_values:
                scene_encoding += f"{{has_attr({oid1}, any, {cleanup_whitespace(standalone_value_)})}}.\n"
                scene_encoding += f":~ has_attr({oid1}, any, {cleanup_whitespace(standalone_value_)}). [{prob_to_asp_weight(standalone_probs[0,k])}, ({oid1}, any, {cleanup_whitespace(standalone_value_)})]\n"
                scene_encoding += f":~ not has_attr({oid1}, any, {cleanup_whitespace(standalone_value_)}). [{prob_to_asp_weight(standalone_probs[1,k])}, ({oid1}, any, {cleanup_whitespace(standalone_value_)})]\n"
                k += 1

            del standalone_scores, standalone_probs

        # get cosine similarities between relations and every object pair's image crop
        if len(relations) > 0 and len(objects) > 1:
            rel_prompts = []
            for oid2, object2 in object_items:
                if oid2 != oid1:
                    for rel in relations:
                        rel_prompts.append(f"{get_article(object1['name'])} {object1['name']} {rel} {get_article(object2['name'])} {object2['name']}")
                    rel_prompts.append(f"{get_article(object1['name'])} {object1['name']} and {get_article(object2['name'])} {object2['name']}")

            rel_logits_per_image = model.score(rel_bbox_crops, rel_prompts)
            
            m = 0
            for o2, (oid2, object2) in enumerate(object_items):
                if oid1 != oid2:
                    rel_scores = torch.stack([
                        rel_logits_per_image[rel_bbox_indices[o1, o2], m*(num_relations+1):m*(num_relations+1)+num_relations],
                        rel_logits_per_image[rel_bbox_indices[o1, o2], m*(num_relations+1)+num_relations].expand(num_relations)
                    ])
                    rel_probs = torch.nn.functional.softmax(rel_scores, dim=0)

                    n = 0
                    for rel in relations:
                        scene_encoding += f"{{has_rel({oid1}, {cleanup_whitespace(rel)}, {oid2})}}.\n"
                        scene_encoding += f":~ has_rel({oid1}, {cleanup_whitespace(rel)}, {oid2}). [{prob_to_asp_weight(rel_probs[0,n])}, ({oid1}, {cleanup_whitespace(rel)}, {oid2})]\n"
                        scene_encoding += f":~ not has_rel({oid1}, {cleanup_whitespace(rel)}, {oid2}). [{prob_to_asp_weight(rel_probs[1,n])}, ({oid1}, {cleanup_whitespace(rel)}, {oid2})]\n"

                        n += 1
                    
                    scene_encoding += "\n"
                    m += 1

                    del rel_scores, rel_probs
                
            del rel_logits_per_image

    try: 
        del obj_logits_per_image
    except:
        pass

    return scene_encoding