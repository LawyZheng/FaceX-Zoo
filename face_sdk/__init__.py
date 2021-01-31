import sys
sys.path.append('.')

import yaml
import cv2
import numpy as np
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler

with open('config/model_conf.yaml') as f:
    model_conf = yaml.load(f)

class FaceFeature:
    def __init__(self, data):
        self.data = data

class FaceRecognize:
    __model_path = "models"

    def __init__(self, scene='non-mask', device='cpu'):
        self.scene = scene
        self.device = device
        self.detectHandler = self.__get_detect_handler()
        self.alignHandler = self.__get_alignment_handler()
        self.recognitionHandler = self.__get_recognition_handler()

    def __handler_helper(self, category, loader, handler):
        model_name =  model_conf[self.scene][category]
        l = loader(self.__model_path, category, model_name)
        model, cfg = l.load_model(self.device)
        return handler(model, self.device, cfg)

    def __get_detect_handler(self):
        return self.__handler_helper('face_detection', FaceDetModelLoader, FaceDetModelHandler)

    def __get_alignment_handler(self):
        return self.__handler_helper('face_alignment', FaceAlignModelLoader, FaceAlignModelHandler)

    def __get_recognition_handler(self):
        return self.__handler_helper('face_recognition', FaceRecModelLoader, FaceRecModelHandler)

    def GetOneFaceFeature(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        face_cropper = FaceRecImageCropper()
        dets = self.detectHandler.inference_on_image(image)

        if dets.shape[0] > 1:
            raise Exception("Detect two faces")

        landmarks = self.alignHandler.inference_on_image(image, dets[0])
        landmarks_list = []
        for (x, y) in landmarks.astype(np.int32):
            landmarks_list.extend((x, y))
        cropped_image = face_cropper.crop_image_by_mat(image, landmarks_list)
        return FaceFeature(self.recognitionHandler.inference_on_image(cropped_image))

    def GetFacesFeauture(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        face_cropper = FaceRecImageCropper()
        dets = self.detectHandler.inference_on_image(image)

        feature_list = list()
        for i in range(dets.shape[0]):
            landmarks = self.alignHandler.inference_on_image(image, dets[i])
            landmarks_list = []
            for (x, y) in landmarks.astype(np.int32):
                landmarks_list.extend((x, y))
            cropped_image = face_cropper.crop_image_by_mat(image, landmarks_list)
            feature_list.append(FaceFeature(self.recognitionHandler.inference_on_image(cropped_image)))
        return feature_list

    def CompareTwoFaces(self, face_one, face_two):
        if not (isinstance(face_one, FaceFeature) and (isinstance(face_two, FaceFeature))):
            raise Exception("Requires FaceFeature Type.")

        return np.dot(face_one.data, face_two.data)

    
if __name__ == '__main__':
    faceRec = FaceRecognize()
    path = "/lawyzheng/FaceX-Zoo/face_sdk/images/emma.jpg"
    source = faceRec.GetOneFaceFeature(path)

    path = "/lawyzheng/FaceX-Zoo/face_sdk/test_images/1.jpeg"
    target = faceRec.GetFacesFeauture(path)

    print(faceRec.CompareTwoFaces(source, target[1]))
