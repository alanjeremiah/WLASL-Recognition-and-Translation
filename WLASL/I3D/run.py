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

import language
from dotenv import load_dotenv

from itertools import chain

import pickle

load_dotenv("posts/nlp/.env", override=True)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)

args = parser.parse_args()

def load_rgb_frames_from_video():
    
    vidcap = cv2.VideoCapture(0)

    
    frames = []
    
    offset = 0
    text = " "
    batch = 40
    text_list = []
    word_list = []
    sentence = ""
    text_count = 0
    
    """
    To maintain the continous flow of actions we bring in the the batch size and offest modulo factor.
    the batch size and the offset can be varied.
    
    """
    
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
                
                if offset % 20 == 0:
                    text = run_on_tensor(torch.from_numpy((np.asarray(frames, dtype=np.float32)).transpose([3, 0, 1, 2])))
                    if text != " ":
                        text_count = text_count + 1
                        
                        if bool(text_list) != False and bool(word_list) != False and text_list[-1] != text and word_list[-1] != text or bool(text_list) == False:
                            text_list.append(text)
                            word_list.append(text)
                            sentence = sentence + " " + text
                            
                        word = language.get_suggestions(text_list, n_gram_counts_list, vocabulary, k = 1.0)
                        if(word != " ."):
                            sentence += word
                            text_list.append(word)
                        
                        if(text_count > 2):
                            sentence = nlp(text_list,**params)
                        cv2.putText(frame1, sentence, (120, 520), font, 0.9, (0, 255, 255), 2, cv2.LINE_4)

                        
            else:
                frames.append(frame)
                if offset == batch:
                    text = run_on_tensor(torch.from_numpy((np.asarray(frames, dtype=np.float32)).transpose([3, 0, 1, 2])))
                    if text != " ":
                        text_count = text_count + 1
                        if bool(text_list) != False and bool(word_list) != False and text_list[-1] != text and word_list[-1] != text or bool(text_list) == False:
                            text_list.append(text)
                            word_list.append(text)
                            sentence = sentence + " " + text

                        word = language.get_suggestions(text_list, n_gram_counts_list, vocabulary, k = 1.0)
                        if(word != " ." ):
                            sentence += word
                            text_list.append(word)
                            
                        if(text_count > 2):
                            sentence = nlp(text_list,**params)
                        cv2.putText(frame1, sentence, (120, 520), font, 0.9, (0, 255, 255), 2, cv2.LINE_4)
                
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            cv2.putText(frame1, sentence, (120, 520), font, 0.9, (0, 255, 255), 2, cv2.LINE_4)
            cv2.imshow('frame', frame1)
            
            if len(text_list) > 10:
                text_list.pop()
                text_list.pop()
                text_list.pop()
            
        else:
            break
            
    vidcap.release()
    cv2.destroyAllWindows()
    


def load_model(weights, num_classes):
    
    #Loading the Inception 3D Model
        
    global i3d 
    i3d = InceptionI3d(400, in_channels=3)

    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights))  # nslt_2000_000700.pt nslt_1000_010800 nslt_300_005100.pt(best_results)  nslt_300_005500.pt(results_reported) nslt_2000_011400
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    i3d.eval()
    
    #Loading the KeytoText model
    
    global nlp
    nlp = pipeline("k2t-new") # The pre-trained models available are 'k2t', 'k2t-base', 'mrm8488/t5-base-finetuned-common_gen', 'k2t-new'
    global params
    params = {"do_sample":True, "num_beams": 5, "no_repeat_ngram_size":2, "early_stopping":True}
    
    #Loading the NGram model
    
    with open("NLP/nlp_data_processed", "rb") as fp:   # Unpickling
           train_data_processed = pickle.load(fp)
    
    global n_gram_counts_list
    with open("NLP/nlp_gram_counts", "rb") as fp:   # Unpickling
        n_gram_counts_list = pickle.load(fp)
        
    global vocabulary
    vocabulary = list(set(chain.from_iterable(train_data_processed)))
    
    
    load_rgb_frames_from_video()
    

def run_on_tensor(ip_tensor):

    ip_tensor = ip_tensor[None, :]
    
    t = ip_tensor.shape[2] 
    ip_tensor.cuda()
    per_frame_logits = i3d(ip_tensor)

    predictions = F.upsample(per_frame_logits, t, mode='linear')

    predictions = predictions.transpose(2, 1)
    out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
    arr = predictions.cpu().detach().numpy()[0] 

    print(float(max(F.softmax(torch.from_numpy(arr[0]), dim=0))))
    print(wlasl_dict[out_labels[0][-1]])
    
    """
    
    The 0.5 is threshold value, it varies if the batch sizes are reduced.
    
    """
    if max(F.softmax(torch.from_numpy(arr[0]), dim=0)) > 0.5:
        return wlasl_dict[out_labels[0][-1]]
    else:
        return " " 
        
    
def create_WLASL_dictionary():
    
    global wlasl_dict 
    wlasl_dict = {}
    
    with open('preprocess/wlasl_class_list.txt') as file:
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
    num_classes = 2000
    save_model = './checkpoints/'

    root = '../../data/WLASL2000'

    train_split = 'preprocess/nslt_{}.json'.format(num_classes)
    weights = 'archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'
    
    
    create_WLASL_dictionary()
    
    load_model(weights, num_classes)
        
    
    