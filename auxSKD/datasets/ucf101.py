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
        self.rgb_lines = list(lines) 
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
        sample_name, num_frames = rgb_line[0], int(rgb_line[1])
        #sample_name, action_label, num_frames = rgb_line[0], int(rgb_line[1]), int(rgb_line[2])

        rgb_dir = os.path.join(self.rgb_prefix, sample_name)
        segment_s = random.randint(1, self.max_segment)
        segment_t = random.randint(1, self.max_segment)

        while segment_s == segment_t:

            segment_t = random.randint(1, self.max_sr)        




        segment_start_frame_s = int((segment_s-1)*(self.clip_len/self.max_segment))
        segment_last_frame_s = int((segment_s)*(self.clip_len/self.max_segment))



        segment_start_frame_t = int((segment_t-1)*(self.clip_len/self.max_segment))
        segment_last_frame_t = int((segment_t)*(self.clip_len/self.max_segment))




        sample_rate_s = random.randint(1, self.max_sr)
       
        start_frame_s = random.randint(1, num_frames-self.clip_len)
       # print(start_frame_s)


        sample_rate_t = random.randint(1, self.max_sr)
        
        start_frame_t = random.randint(1, num_frames-self.clip_len)
        #############################################

        while sample_rate_s == sample_rate_t:

            sample_rate_t = random.randint(1, self.max_sr)


        rgb_clip_s = self.loop_load_rgb(rgb_dir, start_frame_s, sample_rate_s,
                                      self.clip_len, num_frames, segment_start_frame_s, segment_last_frame_s)

        #############################################

        rgb_clip_t = self.loop_load_rgb(rgb_dir, start_frame_t, sample_rate_t,
                                      self.clip_len, num_frames, segment_start_frame_t, segment_last_frame_t)

        ############################################

        label = sample_rate_s - 1

        trans_clip_s = self.transforms_(rgb_clip_s)
        trans_clip_t = self.transforms_(rgb_clip_t)


        ## apply different color jittering for each frame in the video clip
        trans_clip_cj_s = []
        for frame in trans_clip_s:
            frame = self.toPIL(frame)  # PIL image
            frame = self.color_jitter_(frame)  # tensor [C x H x W]
            frame = np.array(frame)
            trans_clip_cj_s.append(frame)

        trans_clip_cj_s = np.array(trans_clip_cj_s).transpose(3, 0, 1, 2)


        trans_clip_cj_t = []
        for frame in trans_clip_t:
            frame = self.toPIL(frame)  # PIL image
            frame = self.color_jitter_(frame)  # tensor [C x H x W]
            frame = np.array(frame)
            trans_clip_cj_t.append(frame)

        trans_clip_cj_t = np.array(trans_clip_cj_t).transpose(3, 0, 1, 2)


        return trans_clip_cj_s, trans_clip_cj_t, label

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

                if (start_frame + (idx1 + 3) * sample_rate) > num_frames:
                    start_frame = 1
                    normal_f = 1
                    idx = 0
                    idx1 = 0
                else:
                    idx1 += 1



                # idx=0
            else:
                # if idx == 0 & i != 0:
                #    normal_f = normal_f + 1 

                cur_img_path = os.path.join(
                    video_dir,
                    "frame" + "{:06}.jpg".format(normal_f + idx))

                start_frame = normal_f + idx
                idx1=1
            

               # print(cur_img_path)

                img = cv2.imread(cur_img_path)
                video_clip.append(img)

                if (normal_f + (idx + 3) ) > num_frames:
                    normal_f = 1
                    start_frame = 1
                    idx1 = 0
                    idx = 0
                else:
                    idx += 1




        video_clip = np.array(video_clip)

        return video_clip

