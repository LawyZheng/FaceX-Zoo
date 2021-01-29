import sys
sys.path.append('.')
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')

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


model_path = 'models'
scene = 'non-mask'

def get_detect_model():
    model_category = 'face_detection'
    model_name =  model_conf[scene][model_category]
    logger.info('Start to load the face detection model...')
    faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
    model, cfg = faceDetModelLoader.load_model()
    faceDetModelHandler = FaceDetModelHandler(model, 'cpu', cfg)
    logger.info('Success!')
    return faceDetModelHandler


def get_alignment_model():
    model_category = 'face_alignment'
    model_name =  model_conf[scene][model_category]
    logger.info('Start to load the face landmark model...')
    faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
    model, cfg = faceAlignModelLoader.load_model()
    faceAlignModelHandler = FaceAlignModelHandler(model, 'cpu', cfg)
    logger.info('Success!')
    return faceAlignModelHandler

def get_recognition_model():
    model_category = 'face_recognition'
    model_name =  model_conf[scene][model_category]    
    logger.info('Start to load the face recognition model...')
    faceRecModelLoader = FaceRecModelLoader(model_path, model_category, model_name)
    model, cfg = faceRecModelLoader.load_model()
    faceRecModelHandler = FaceRecModelHandler(model, 'cpu', cfg)
    logger.info('Success!')
    return faceRecModelHandler

def get_face_feature_list(path, det_model, alig_model, rec_model):
    '''
    return: type -> list, face feature
    '''
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    face_cropper = FaceRecImageCropper()
    dets = det_model.inference_on_image(image)
    feature_list = []
    for i in range(dets.shape[0]):
        landmarks = alig_model.inference_on_image(image, dets[i])
        landmarks_list = []
        for (x, y) in landmarks.astype(np.int32):
            landmarks_list.extend((x, y))
        cropped_image = face_cropper.crop_image_by_mat(image, landmarks_list)
        feature = rec_model.inference_on_image(cropped_image)
        feature_list.append(feature)

    return feature_list

if __name__ == '__main__':
    import os
    det_model = get_detect_model()
    alig_model = get_alignment_model() 
    rec_model = get_recognition_model()

    source_map = dict()

    path = "/lawyzheng/FaceX-Zoo/face_sdk/images/"
    for each_source in os.listdir(path):
        source_map[each_source.split(".")[0]] = get_face_feature_list(path+each_source, det_model, alig_model, rec_model)[0]
        # source_list += get_face_feature_list(path+each_source, det_model, alig_model, rec_model)
        # name_list.append(each_source.split(".")[0])
    
    target_path = '/lawyzheng/FaceX-Zoo/face_sdk/test_images/10.jpg'
    target_feature_list = get_face_feature_list(target_path, det_model, alig_model, rec_model)
    
    result = list()
    for each_target in target_feature_list:
        tmp_name = ""
        tmp_score = 0
        for name, feature in source_map.items():
            score = np.dot(each_target, feature)
            print("name: %s, score: %.2f" % (name, score))
            if score > tmp_score:
                tmp_score = score
                tmp_name = name

        if tmp_score > 0.55:
            result.append(tmp_name)

        print('====' * 10)

    print("result: ", result)
    













