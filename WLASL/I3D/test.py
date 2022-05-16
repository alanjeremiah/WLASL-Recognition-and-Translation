"""

This is a test page, the run.py file is the main running program

"""
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 20:10:06 2022

@author: 24412
"""
import math
import os
import argparse

import matplotlib.pyplot as plt

import torch
import torch.nn as nn


import numpy as np

import torch.nn.functional as F
from pytorch_i3d import InceptionI3d

import cv2
from keytotext import pipeline



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)

args = parser.parse_args()

def load_rgb_frames_from_video():
    
    #video_path = 'D:/Project/WLASL/data/Phrases/Wouldyouliketodance.mp4'
    #video_path = 'D:/Project/WLASL/data/WLASL2000/63233.mp4'
    
    #target = open('D:\Project\WLASL\code\I3D\preprocess\wlasl_2000_frame_45.txt','w')
    #count = 0;
    
    """for key in wlasl_test_dict.keys():
        video_path = 'D:/Project/WLASL/data/WLASL2000/'
        video_path = os.path.join(video_path, key + '.mp4')"""
    
    
    vidcap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    # vidcap = cv2.VideoCapture('/home/dxli/Desktop/dm_256.mp4')
    
    #vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    #print(num)
    
    frames = []
    
    
    #print(vidcap.isOpened())
    offset = 0
    text = " "
    batch = 35
    text_list = []
    sentence = ""
    
    while True:
        ret, frame1 = vidcap.read()
        offset = offset + 1
        font = cv2.FONT_HERSHEY_TRIPLEX
        
        if ret == True:
            
            w, h, c = frame1.shape
            sc = 224 / w
            sx = 224 / h
            frame = cv2.resize(frame1, dsize=(0, 0), fx=sx, fy=sc)
            frame1 = cv2.resize(frame1, dsize = (1280,720))
    
            frame = (frame / 255.) * 2 - 1
            
            if offset > batch:
                frames.pop(0)
                frames.append(frame)
                
                if offset % 10 == 0:
                    text = run_on_tensor(torch.from_numpy((np.asarray(frames, dtype=np.float32)).transpose([3, 0, 1, 2])))
                    if text != " ":
                        """ ngram <- second iter < check the last in text_list, before ngram generation"""
                        if bool(text_list) != False and text_list[-1] != text or bool(text_list) == False:
                            text_list.append(text)
                            sentence = "The problem as mentioned above is caused by the camera driver. I was able to fix it using Direct Show as a backend. I read (sorry, but I do not remember where) that almost all cameras provide a driver that allows their use from DirectShow. Therefore, I used DirectShow in Windows to interact with the cameras and I was able to configure the resolution as I wanted and also get the native aspect ratio of my camera (16: 9). You can try this code to see if this works for you."
                    if len(text_list) < 3:
                        cv2.putText(frame1, sentence, (120, 520), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
                    else:
                        #sentence = nlp(text_list)
                        cv2.putText(frame1, sentence, (120, 520), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
                        
                        
                    """if text != " ":
                        count = count + 1
                        target.write('%d\t%d\t%s\t%s\n' % (count, num, key, wlasl_test_dict[key]))"""
                        
            else:
                frames.append(frame)
                if offset == batch:
                    text = run_on_tensor(torch.from_numpy((np.asarray(frames, dtype=np.float32)).transpose([3, 0, 1, 2])))
                    if text != " ":
                        if bool(text_list) != False and text_list[-1] != text or bool(text_list) == False:
                            text_list.append(text)
                            sentence = "The problem as mentioned above is caused by the camera driver. I was able to fix it using Direct Show as a backend. I read (sorry, but I do not remember where) that almost all cameras provide a driver that allows their use from DirectShow. Therefore, I used DirectShow in Windows to interact with the cameras and I was able to configure the resolution as I wanted and also get the native aspect ratio of my camera (16: 9). You can try this code to see if this works for you."
                    if len(text_list) < 3:
                        cv2.putText(frame1, sentence, (120, 520), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
                    else:
                        #sentence = nlp(text_list)
                        cv2.putText(frame1, sentence, (120, 520), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
                            
                        
                        """if text != " ":
                            count = count + 1
                            target.write('%d\t%d\t%s\t%s\n' % (count, num, key, wlasl_test_dict[key]))"""
                        
                    #frames.clear()
                    #count = 0
                
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            cv2.putText(frame1, sentence, (120, 520), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('frame', frame1)
            
            if len(text_list) == 5:
                text_list.clear()
                sentence = ""
        else:
            break
            
        
        
    
    vidcap.release()
    cv2.destroyAllWindows()
    


def load_model(weights, num_classes):
    global i3d 
    i3d = InceptionI3d(400, in_channels=3)
    # i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))

    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights))  # nslt_2000_000700.pt nslt_1000_010800 nslt_300_005100.pt(best_results)  nslt_300_005500.pt(results_reported) nslt_2000_011400
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()
    
    #global nlp
    #nlp = pipeline("k2t-new")
    
    load_rgb_frames_from_video()
    
    
def test_load_model(weights, num_classes):
    global i3d 
    i3d = InceptionI3d(400, in_channels=3)
    # i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))

    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights))  # nslt_2000_000700.pt nslt_1000_010800 nslt_300_005100.pt(best_results)  nslt_300_005500.pt(results_reported) nslt_2000_011400
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()
    
    
    create_test_list()

def run_on_tensor(ip_tensor):

    ip_tensor = ip_tensor[None, :]
    
    t = ip_tensor.shape[2] 
    ip_tensor.cuda()
    per_frame_logits = i3d(ip_tensor)

    predictions = F.upsample(per_frame_logits, t, mode='linear')

    predictions = predictions.transpose(2, 1)
    out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
    out_probs = np.sort(predictions.cpu().detach().numpy()[0])

    arr = predictions.cpu().detach().numpy()[0] #[0,:,0].T
    
    
    #plt.plot(range(len(arr[0])), F.softmax(torch.from_numpy(arr[0]), dim=0).numpy())
    #plt.show()
    
    print(float(max(F.softmax(torch.from_numpy(arr[0]), dim=0))))
    #F.softmax(torch.from_numpy(arr[0])
    print(wlasl_dict[out_labels[0][-1]])
    
    #target = open('D:\Project\WLASL\code\I3D\preprocess\wlasl_100_frame_45.txt','w')
    
    if max(F.softmax(torch.from_numpy(arr[0]), dim=0)) > 0.5:
        return wlasl_dict[out_labels[0][-1]]
    else:
        return " " 
    

def test_run_on_tensor(weights, ip_tensor, num_classes):
    i3d = InceptionI3d(400, in_channels=3)
    # i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))

    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights))  # nslt_2000_000700.pt nslt_1000_010800 nslt_300_005100.pt(best_results)  nslt_300_005500.pt(results_reported) nslt_2000_011400
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()

    t = ip_tensor.shape[2]
    ip_tensor.cuda()
    per_frame_logits = i3d(ip_tensor)

    predictions = F.upsample(per_frame_logits, t, mode='linear')

    predictions = predictions.transpose(2, 1)
    out_labels = np.argsort(predictions.cpu().detach().numpy()[0])

    arr = predictions.cpu().detach().numpy()[0,:,0].T

    plt.plot(range(len(arr)), F.softmax(torch.from_numpy(arr), dim=0).numpy())
    plt.show()
    
    
    #print(out_labels)

    #return out_labels
    
def create_test_list():
    global wlasl_test_dict
    wlasl_test_dict = {}
    
    with open('D:\Project\WLASL\code\I3D\preprocess\wlasl_2000_list.txt') as file:
        for line in file:
            split_list = line.split()
            key = str(split_list[2])
            value = split_list[1]
            wlasl_test_dict[key] = value
        
    load_rgb_frames_from_video()
    
def create_WLASL_dictionary():
    
    global wlasl_dict 
    wlasl_dict = {}
    
    with open('D:\Project\WLASL\code\I3D\preprocess\wlasl_class_list.txt') as file:
        for line in file:
            split_list = line.split()
            if len(split_list) != 2:
                key = int(split_list[0])
                value = split_list[1] + " " + split_list[2]
            else:
                key = int(split_list[0])
                value = split_list[1]
            wlasl_dict[key] = value
            

if __name__ == '__main__':
    # ================== test i3d on a dataset ==============
    # need to add argparse
    mode = 'rgb'
    num_classes = 100
    save_model = './checkpoints/'

    root = '../../data/WLASL2000'

    train_split = 'preprocess/nslt_{}.json'.format(num_classes)
    weights = 'archived/asl100/FINAL_nslt_100_iters=896_top1=65.89_top5=84.11_top10=89.92.pt'
    
    #create_test_list()
    
    create_WLASL_dictionary()
    
    load_model(weights, num_classes)
    
    #load_rgb_frames_from_video()
    

    #run(mode=mode, root=root, save_model=save_model, train_split=train_split, weights=weights)