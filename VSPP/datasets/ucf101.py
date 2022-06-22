#VSPP

import os
import cv2
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils.video_transforms import RandomCrop, RandomHorizontalFlip, CenterCrop, ClipResize, ToTensor
import copy 
class ucf101_pace_pretrain(Dataset):

    def __init__(self, data_list, rgb_prefix, clip_len,  max_sr, max_segment, transforms_=None, color_jitter_=None): # yapf: disable:
        lines = open(data_list)
        self.rgb_lines = list(lines) * 10
        self.rgb_prefix = rgb_prefix
        self.clip_len = clip_len
        self.max_sr = max_sr
        self.toPIL = transforms.ToPILImage()
        self.transforms_ = transforms_
        self.color_jitter_ = color_jitter_
        self.max_segment = max_segment

    def __len__(self):
        return len(self.rgb_lines)

    def __getitem__(self, idx):
        rgb_line = self.rgb_lines[idx].strip('\n').split()
        sample_name, action_label, num_frames = rgb_line[0], int(rgb_line[1]), int(rgb_line[2])
        #sample_name, num_frames = rgb_line[0], int(rgb_line[1])

        rgb_dir = os.path.join(self.rgb_prefix, sample_name)
        sample_rate = random.randint(1, self.max_sr)

        segment = random.randint(1, self.max_segment)        
        start_frame = random.randint(1, num_frames-self.clip_len)
        # print("len {}, start_frame {} ".format(num_frames, start_frame))


        segment_start_frame = int((segment-1)*(self.clip_len/self.max_segment))
        segment_last_frame = int((segment)*(self.clip_len/self.max_segment))
        # print("sample_rate {} , segment: {}".format(sample_rate, segment))
        # print("segment_start_frame {} , segment_last_frame: {}".format(segment_start_frame, segment_last_frame))






        rgb_clip = self.loop_load_rgb(rgb_dir, start_frame, sample_rate,
                                      self.clip_len, num_frames, segment_start_frame, segment_last_frame)

        label_speed = sample_rate - 1
        label_segment = segment - 1
        label = [label_speed, label_segment]

        trans_clip = self.transforms_(rgb_clip)

        ## apply different color jittering for each frame in the video clip
        trans_clip_cj = []
        for frame in trans_clip:
            frame = self.toPIL(frame)  # PIL image
            frame = self.color_jitter_(frame)  # tensor [C x H x W]
            frame = np.array(frame)
            trans_clip_cj.append(frame)

        trans_clip_cj = np.array(trans_clip_cj).transpose(3, 0, 1, 2)

        return trans_clip_cj, np.array(label)

    def loop_load_rgb(self, video_dir, start_frame, sample_rate, clip_len,
                      num_frames, segment_start_frame, segment_last_frame):

        video_clip = []
        idx1 = 0
        idx = 0
        normal_f=copy.deepcopy(start_frame)

        for i in range(clip_len):
            if segment_start_frame <= i <= segment_last_frame:

                
                cur_img_path = os.path.join(
                    video_dir,
                    "frame" + "{:06}.jpg".format(start_frame + idx1 * sample_rate))
                normal_f = (start_frame + (idx1 * sample_rate))
                idx = 1


                # print(cur_img_path)

                img = cv2.imread(cur_img_path)
                video_clip.append(img)

                if (start_frame + (idx1 + 1) * sample_rate) > num_frames:
                    start_frame = 1
                    normal_f = 1
                    idx = 0
                    idx1 = 0
                else:
                    idx1 += 1



                # idx=0
            else:


                cur_img_path = os.path.join(
                    video_dir,
                    "frame" + "{:06}.jpg".format(normal_f + idx))

                start_frame = normal_f + idx
                idx1=1
            

                # print(cur_img_path)

                img = cv2.imread(cur_img_path)
                video_clip.append(img)

                if (normal_f + (idx + 1) ) > num_frames:
                    normal_f = 1
                    start_frame = 1
                    idx1 = 0
                    idx = 0
                else:
                    idx += 1




        video_clip = np.array(video_clip)

        return video_clip
