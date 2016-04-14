#!/usr/bin/env python

import sys
import numpy as np
import cv2
import sys
from glob import glob
import itertools as it
import matplotlib.pyplot as plt

class hog_detector():

    def __init__(self, input_video, output, skip_every):

        self.video_source = 0 if input_video == None else input_video	        
	self.video_output = 'default.avi' if output == None else output+'.avi'
        self.skip_every = 4 if skip_every == None else 4


    def inside(self, r, q):
        rx, ry, rw, rh = r
        qx, qy, qw, qh = q
        return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

    
    def draw_detections(self, img, rects, thickness = 1):
        for x, y, w, h in rects:
            # the HOG detector returns slightly larger rectangles than the real objects.
            # so we slightly shrink the rectangles to get a nicer output.
            pad_w, pad_h = int(0.15*w), int(0.05*h)
            cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


    def detect_video(self):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
          
        # Iniciar entrada de video
        video_capture = cv2.VideoCapture(self.video_source)
        video_capture.set(cv2.cv.CV_CAP_PROP_FPS, 5) 
        # Iniciar salida de video
        ret, frame = video_capture.read()
	height, width, layers = frame.shape 
        fps=5
        fourcc = cv2.cv.CV_FOURCC('M','J','P','G') 
        out = self.video_output
	video = cv2.VideoWriter(out, fourcc, fps, (width, height), 1)        
        # Skip frames count
        count_skip = 0        

        while True:
            count_skip += 1
            
            # Capturar la entrada de video
            ret, frame = video_capture.read()
            if frame is None:
                break
            if count_skip<self.skip_every: 
                continue
            count_skip=0
            
            # Detectar en la imagen
            found, w = hog.detectMultiScale(frame, winStride=(8,8), padding=(16,16), scale=1.05)
            found_filtered = []
            for ri, r in enumerate(found):
                for qi, q in enumerate(found):
                    if ri != qi and self.inside(r, q):
                        break
                    else:
                        found_filtered.append(r)
            # Dibujar detecciones
            self.draw_detections(frame, found)
            self.draw_detections(frame, found_filtered, 3)
                
            # Guardar las detecciones. 
            # salida en video con detecciones  
            video.write(frame)             


            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

	cv2.destroyAllWindows()
        video_capture.release()
        video.release()  
  
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input video, leave emtpy to use cam")
    parser.add_argument("--output", help="where to save video")
    parser.add_argument("--skip", help="frames to skip")
    args = parser.parse_args()

    x = hog_detector(args.input, args.output, args.skip)
    x.detect_video()
    


