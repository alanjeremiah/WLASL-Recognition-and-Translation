"""

This code is to create a folder containing only one video for a word.
The purpose of this folder is to be able to practice the ASL for demo purposes.

"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 20:54:40 2022

@author: 24412
"""

import json
import os
import os.path
import shutil


def WLASLN(split_file, root):
    dictionary = {}
    vid_id = []
    label_id = set()
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    for vid in data.keys():
        if data[vid]['subset'] != "test":
            continue

        x = data[vid]['action'][0]
        if x in label_id:
            continue
        
        label_id.add(x)
        vid_id.append(vid)
        
        dictionary[x] = vid
        i += 1
        
        
    create_WLASLN_folder(vid_id, root)
    create_WLASLN_list(dictionary)

    print(len(dictionary))
    
    
def create_WLASLN_list(dictionary):
    
    
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
           
    target = open('preprocess/wlasl_2000_list.txt','w')
    keys = dictionary.keys()
    for key in keys:
        target.write('%s\t%s\t%s\n' % (key, wlasl_dict[key], dictionary[key]))
        
    
    target.close()



def create_WLASLN_folder(video_id, root):
    
    target_root = r'../../data/WLASL100'  #Change the folder name to WLASL100, WLASL300, WLASL1000 or WLASL2000 based on the num_classes
    
    for vid in video_id:        
        video_path = os.path.join(root, vid + '.mp4')
        if not os.path.exists(video_path):
            continue
        
        target = os.path.join(target_root, vid + '.mp4')
        if not os.path.exists(target):
            shutil.copyfile(video_path, target)


if __name__ == '__main__':
    
    num_classes = 2000
    root = '../../data/WLASL2000'
    
    train_split = 'preprocess/nslt_{}.json'.format(num_classes)
    WLASLN(train_split, root)