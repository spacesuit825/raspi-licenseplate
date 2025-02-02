## Code to train and test the ANPR model.
from tflite_support import metadata
import numpy as np
import os
from picamera.array import PiRGBArray
from picamera import PiCamera
from datetime import datetime
import time

import tflite_runtime.interpreter as tflite

from typing import List, NamedTuple
import json
import numpy as np
from PIL import Image

import os
from scipy import ndimage
import scipy
from scipy.spatial import distance as dist
from scipy.stats import mode
from collections import OrderedDict
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

import pytesseract
import cv2
import jellyfish
#import tqdm

print("Loading model...")
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
print('Tesseract Engine:', pytesseract.get_tesseract_version())

'''class Train:
    def __init__(self, train_path, val_path):
        train_data, val_data = self.load_data(train_path, val_path)
        self.training(train_data, val_data)

    def load_data(self, train_path, val_path):
        train_data = object_detector.DataLoader.from_pascal_voc(
            train_path,
            train_path,
            ['vehicle', 'license-plate']
        )

        val_data = object_detector.DataLoader.from_pascal_voc(
            val_path,
            val_path,
            ['vehicle', 'license-plate']
        )
        return train_data, val_data
    
    def training(self, train_data, val_data):
        spec = model_spec.get('efficientdet_lite2')

        model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=40, validation_data=val_data)
        model.export(export_dir='.', tflite_filename='android.tflite')

        model.evaluate_tflite('android.tflite', val_data)'''

## Inference Code. This is the stuff that'll be on the RPi. ##

Interpreter = tflite.Interpreter

class ObjectDetectorOptions(NamedTuple):

  enable_edgetpu: bool = False

  label_allow_list: List[str] = None

  label_deny_list: List[str] = None

  max_results: int = -1

  num_threads: int = 1

  score_threshold: float = 0.0


class Rect(NamedTuple):
  left: float
  top: float
  right: float
  bottom: float


class Category(NamedTuple):
  label: str
  score: float
  index: int


class Detection(NamedTuple):
  bounding_box: Rect
  categories: List[Category]

class ObjectDetector:
  _OUTPUT_LOCATION_NAME = 'location'
  _OUTPUT_CATEGORY_NAME = 'category'
  _OUTPUT_SCORE_NAME = 'score'
  _OUTPUT_NUMBER_NAME = 'number of detections'

  def __init__(
      self,
      model_path: str,
      options: ObjectDetectorOptions = ObjectDetectorOptions()
    ):

    # Load metadata from model.
    displayer = metadata.MetadataDisplayer.with_model_file(model_path)

    model_metadata = json.loads(displayer.get_metadata_json())
    process_units = model_metadata['subgraph_metadata'][0]['input_tensor_metadata'][0]['process_units']
    mean = 0.0
    std = 1.0
    for option in process_units:
      if option['options_type'] == 'NormalizationOptions':
        mean = option['options']['mean'][0]
        std = option['options']['std'][0]
    self._mean = mean
    self._std = std

    file_name = displayer.get_packed_associated_file_list()[0]
    label_map_file = displayer.get_associated_file_buffer(file_name).decode()
    label_list = list(filter(lambda x: len(x) > 0, label_map_file.splitlines()))
    self._label_list = label_list

    # Initialize TFLite model.
    interpreter = Interpreter(
        model_path=model_path, num_threads=options.num_threads)

    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]

    sorted_output_indices = sorted(
        [output['index'] for output in interpreter.get_output_details()])
    self._output_indices = {
        self._OUTPUT_LOCATION_NAME: sorted_output_indices[0],
        self._OUTPUT_CATEGORY_NAME: sorted_output_indices[1],
        self._OUTPUT_SCORE_NAME: sorted_output_indices[2],
        self._OUTPUT_NUMBER_NAME: sorted_output_indices[3],
    }

    self._input_size = input_detail['shape'][2], input_detail['shape'][1]
    self._is_quantized_input = input_detail['dtype'] == np.uint8
    self._interpreter = interpreter
    self._options = options

  def detect(self, input_image: np.ndarray):
    image_height, image_width, _ = input_image.shape

    input_tensor = self._preprocess(input_image)

    self._set_input_tensor(input_tensor)
    self._interpreter.invoke()

    boxes = self._get_output_tensor(self._OUTPUT_LOCATION_NAME)
    classes = self._get_output_tensor(self._OUTPUT_CATEGORY_NAME)
    scores = self._get_output_tensor(self._OUTPUT_SCORE_NAME)
    count = int(self._get_output_tensor(self._OUTPUT_NUMBER_NAME))

    return self._postprocess(boxes, classes, scores, count, image_width,
                             image_height)

  def _preprocess(self, input_image: np.ndarray):
    input_tensor = cv2.resize(input_image, self._input_size)

    # Normalize the input if it's a float model. This should accelerate model inference.
    if not self._is_quantized_input:
      input_tensor = (np.float32(input_tensor) - self._mean) / self._std

    # Add batch dimension
    input_tensor = np.expand_dims(input_tensor, axis=0)

    return input_tensor

  def _set_input_tensor(self, image):
    tensor_index = self._interpreter.get_input_details()[0]['index']
    input_tensor = self._interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

  def _get_output_tensor(self, name):
    """Returns the output tensor at the given index."""
    output_index = self._output_indices[name]
    tensor = np.squeeze(self._interpreter.get_tensor(output_index))
    return tensor

  def _postprocess(self, boxes: np.ndarray, classes: np.ndarray,
                   scores: np.ndarray, count: int, image_width: int,
                   image_height: int) -> List[Detection]:
    results = []
    for i in range(count):
      if scores[i] >= self._options.score_threshold:
        y_min, x_min, y_max, x_max = boxes[i]
        bounding_box = Rect(
            top=int(y_min * image_height),
            left=int(x_min * image_width),
            bottom=int(y_max * image_height),
            right=int(x_max * image_width))
        class_id = int(classes[i])
        category = Category(
            score=scores[i],
            label=self._label_list[class_id],  # 0 is for background
            index=class_id)
        result = Detection(bounding_box=bounding_box, categories=[category])
        results.append(result)

    # Sort detection results by score ascending
    sorted_results = sorted(
        results,
        key=lambda detection: detection.categories[0].score,
        reverse=True)

    # Filter out detections in deny list
    filtered_results = sorted_results
    if self._options.label_deny_list is not None:
      filtered_results = list(
          filter(
              lambda detection: detection.categories[0].label not in self.
              _options.label_deny_list, filtered_results))

    # Keep only detections in allow list
    if self._options.label_allow_list is not None:
      filtered_results = list(
          filter(
              lambda detection: detection.categories[0].label in self._options.
              label_allow_list, filtered_results))

    if self._options.max_results > 0:
      result_count = min(len(filtered_results), self._options.max_results)
      filtered_results = filtered_results[:result_count]

    return filtered_results


_MARGIN = 10
_ROW_SIZE = 10
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)


def visualize(image: np.ndarray, detections: List[Detection]):
    plate_list = []
    box_list = []
    for detection in detections:
      # Draw bounding_box
      start_point = detection.bounding_box.left, detection.bounding_box.top
      end_point = detection.bounding_box.right, detection.bounding_box.bottom
      cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

      # Draw label and score
      category = detection.categories[0]
      class_name = category.label
      
      plate = None
      if class_name == 'license-plate':
        plate = image.copy()
        plate = plate[detection.bounding_box.top:detection.bounding_box.bottom, detection.bounding_box.left:detection.bounding_box.right]
        box_list.append((detection.bounding_box.left, detection.bounding_box.top, 
                        detection.bounding_box.right, detection.bounding_box.bottom))
        plate_list.append(plate)
      probability = round(category.score, 2)
      result_text = class_name + ' (' + str(probability) + ')'
      text_location = (_MARGIN + detection.bounding_box.left,
                      _MARGIN + _ROW_SIZE + detection.bounding_box.top)
      cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                  _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

    return image, plate_list, box_list

def get_contour_areas(contours):

    all_areas= []

    for cnt in contours:
        area= cv2.contourArea(cnt)
        all_areas.append(area)

    return all_areas

def detect_tilt(dist):
    edge_contours, hierarchy = cv2.findContours(dist, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    recs = []
    areas = []

    max_area = 0
    c = None
    print(len(edge_contours))
    for con in edge_contours:
        x, y, w, h = cv2.boundingRect(con)
        if (w > (1.5 * h)):
            area = w * h
            if area > max_area:
                max_area = area
                c = con

    if c is None:
        lst = sorted(edge_contours, key=cv2.contourArea, reverse=True)
        if lst is not None:
            c = lst[0]

    x, y, w, h = cv2.boundingRect(c)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)

    y_srted = box[np.argsort(box[:, 1]), :]
    tp_c = np.sort(y_srted[2:], axis=0)

    angle = np.rad2deg(np.arctan2(tp_c[1][1] - tp_c[0][1], tp_c[1][0] - tp_c[0][0]))

    if y_srted[2:][0][0] > y_srted[2:][1][0]:
        angle = - angle

    return angle, c

def trim(image):
    np.seterr(divide='ignore', invalid='ignore')

    cols_mean = np.mean(image, axis=0)
    rows_mean = np.mean(image, axis=1)

    row_thres = 10
    col_thres = 10

    border_rows = np.where(rows_mean < row_thres)
    border_cols = np.where(cols_mean < col_thres)

    image[[border_rows], :] = 255
    image[:, [border_cols]] = 255

    return image

def draw_contour_clr(image, contours):
    temp = image.copy()
    for con in contours:
        clr = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        rect = cv2.minAreaRect(con)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(temp, [box], -1, clr, 2)
    return temp

def select_letters(contours, horizon):
    middle_contours = []
    thres = 100
    for con in contours:
        intersections = len(set(range(horizon - thres, horizon + thres)).intersection(con.take(1, 2).flatten()))
        if intersections > 0:
            middle_contours.append(con)
    return middle_contours

'''def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img'''

def skew_correction(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = ndimage.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    print(f'Rotated image by {best_angle}')
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
          borderMode=cv2.BORDER_REPLICATE)

    return rotated

def process_image(image):
    image = cv2.resize(image, (200, 100))
    image = cv2.GaussianBlur(image,(1,1),cv2.BORDER_DEFAULT)
    greyimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    greyimage = skew_correction(greyimage)
    #image = increase_brightness(image, 10)
    sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, sharpen_filter)
    
    return greyimage

    '''
    dst_RGB = cv2.Canny(image, 50, 200)
    ret, thres = cv2.threshold(greyImage, 50, 255, 0)
    ret, thresbin = cv2.threshold(greyImage, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    dist = cv2.Canny(thres, 50, 200)

    ret, threscanny = cv2.threshold(greyImage, 50, 255, 0)

    angle, c = detect_tilt(threscanny)

    rotate_original_img = ndimage.rotate(image.copy(), angle, cval=255)
    rotate_img = ndimage.rotate(image, angle, cval=255)
    rotate_thres = ndimage.rotate(thres.copy(), angle, cval=255)

    cv2.drawContours(image, [c], -1, (0, 0, 255), 2)

    idx = np.where(np.all(rotate_img == (0,255,0), axis=-1))
    if len(idx[0]) > 0:
        max_y = np.max(idx[0])
        min_y = np.min(idx[0])
        max_x = np.max(idx[1])
        min_x = np.min(idx[1])
        w = max_x - min_x
        h = max_y - min_y
        x = min_x
        y = min_y

        cropped_rotate_thres = rotate_thres.copy()[y:y+h, x:x+w]
        cropped_rotate_img = rotate_original_img.copy()[y:y+h, x:x+w]
        greyImage = cv2.cvtColor(cropped_rotate_img, cv2.COLOR_BGR2GRAY)
        ret, thres = cv2.threshold(greyImage, 127, 255, 0)
    
    reduced_thres = trim(thres.copy())
    dist = cv2.Canny(cv2.bitwise_not(reduced_thres), 50, 150)

    height = dist.shape[0]
    bottom = dist[height - int(height/5):-1, :]

    if np.sum(bottom) > 0:
        kernel = np.ones((3, 1), np.uint8)
        opened = cv2.morphologyEx(bottom, cv2.MORPH_OPEN, kernel)
        opened_dist = cv2.Canny(opened, 50, 150)
        dist[height - int(height/5):-1, :] = opened_dist

    contours_upped, hierarchy = cv2.findContours(dist, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    horizon = int(reduced_thres.shape[0]/2)
    contours_middle = select_letters(contours_upped, horizon)

    stencil = np.ones( ( int(reduced_thres.shape[0] * 1.25), int(reduced_thres.shape[1] * 1.25))  , dtype=np.uint8)*255

    for con in contours_middle:
        box = cv2.boundingRect(con)
        x, y, w, h = box
        stencil[y:y+h, x:x+w] = reduced_thres[y:y+h, x:x+w]
    
    cv2.imwrite('num.jpg', stencil)
    return stencil'''

def write_to_json(now, plate_number):
    if not os.path.isfile('plates.json'):
        with open('plates.json', 'w') as f:
            json.dump({}, f)
    with open('plates.json', 'r') as f:
        data = json.load(f)
    data[now] = plate_number
    with open('plates.json', 'w') as f:
        json.dump(data, f)

def read_whitelist():
    with open('whitelist', 'r') as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

class Tracking:
    def __init__(self):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.object_strings = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = 50 # frames until the object is saved.

    def register_object(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.object_strings[self.next_object_id] = []
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister_object(self, object_ID):
        del self.objects[object_ID]
        del self.object_strings[object_ID]
        del self.disappeared[object_ID]

    def str_compile(self, object, plate_number):
        object_id = object[0]
        if len(plate_number) == 6:
            self.object_strings[object_id].append(plate_number)
        else:
            return ''
        if len(self.object_strings[object_id]) > 0:
            print(self.object_strings[object_id])
            return self.frequency_analysis(object_id)
        else:
            return ''
    
    def frequency_analysis(self, object_id):
        lst = self.object_strings[object_id].copy()
        potential_plate = ''.join(mode(i).mode[0] for i in list(zip(*lst)))
        return potential_plate

    def update_between_frames(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister_object(object_id)
            return self.objects
        
        input_centres = np.zeros((len(rects), 2), dtype='int')
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            c_x = int((startX + endX) / 2)
            c_y = int((startY + endY) / 2)
            input_centres[i] = (c_x, c_y)
        
        if len(self.objects) == 0:
            for i in range(0, len(input_centres)):
                self.register_object(input_centres[i])

        else:
            object_ids = list(self.objects.keys())
            object_centres = list(self.objects.values())

            dst = dist.cdist(np.array(object_centres), input_centres)

            rows = dst.min(axis=1).argsort()
            cols = dst.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = input_centres[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

                unused_rows = set(range(0, dst.shape[0])).difference(used_rows)
                unused_cols = set(range(0, dst.shape[1])).difference(used_cols)

                if dst.shape[0] >= dst.shape[1]:
                    for row in unused_rows:
                        object_id = object_ids[row]
                        self.disappeared[object_id] += 1

                        if self.disappeared[object_id] > self.max_disappeared:
                            self.deregister_object(object_id)

                else:
                    for cols in unused_cols:
                        self.register_object(input_centres[cols])

            return self.objects

def jaro_winkler_dist(whitelist, plate_number):
    threshold = 0.8
    for i in whitelist:
        metric = jellyfish.jaro_distance(i, plate_number)
        if metric >= 0.8:
            return i, metric
    return plate_number, 0

def test():
    print('Running detection now...')
    camera = PiCamera()
    camera.resolution = (1280, 720)
    camera.zoom = (0.25,0.25,0.5,0.5)
    rawCapture = PiRGBArray(camera, size=(1280, 720))

    time.sleep(0.5)

    DETECTION_THRESHOLD = 0.4
    TFLITE_MODEL_PATH = "./android.tflite"

    options = ObjectDetectorOptions(
            num_threads=4,
            score_threshold=DETECTION_THRESHOLD,
        )

    detector = ObjectDetector(model_path=TFLITE_MODEL_PATH, options=options)

    whitelist = read_whitelist()

    tracker = Tracking()

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        frame = frame.array.copy()
        frame = np.array(frame)
        frame = Image.fromarray(frame)
        frame.thumbnail((512, 512), Image.ANTIALIAS)
        image_np = np.asarray(frame)

        # Load the TFLite model

        detections = detector.detect(image_np)

        image_np, plate_ls, box_ls = visualize(image_np, detections)

        objects = tracker.update_between_frames(box_ls)
        if objects is not None:
            for (object_id, centre) in objects.items():
                text = 'ID {}'.format(object_id)
                cv2.putText(image_np, text, (centre[0] - 10, centre[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for n, pl in enumerate(plate_ls):
            if pl.size != 0:
                print('Plate detected... ')

                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                potential_plate = ''


                processed_img = process_image(pl)

                cv2.imwrite('hello.jpg', processed_img)
                
                tess_config = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 13 --oem 1 -c tessedit_do_invert=0 -c page_separator="""

                plate_number = pytesseract.image_to_string(processed_img, config = tess_config)

                plate_number = plate_number.strip('\n')
                if objects is not None:
                    if plate_number != '':
                        potential_plate = tracker.str_compile(list(objects.items())[n], plate_number)

                if potential_plate != '':
                    potential_plate, metric = jaro_winkler_dist(whitelist, potential_plate)
                    write_to_json(now, potential_plate)
                    if metric >= 0.8:
                        print(f'The plate {potential_plate} was detected! It matches a whitelisted plate with a confidence of {metric*100}%.')
                    else:
                        print(f'The plate {potential_plate} was detected! It does not match a whitelisted plate, but this may be an error',
                            'so an image of the car has been uploaded.')

                if potential_plate != '':
                    cv2.imwrite(f'{now}_full_image.jpg', image_np)
                    cv2.imwrite(f'{now}.jpg', pl)

        rawCapture.truncate(0)
        #time.sleep(0.5)

test()