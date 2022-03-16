import cv2
from PIL import Image
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--depth_dir', type=str, default=None)
parser.add_argument('--color_dir', type=str, default=None)
args = parser.parse_args()


def depth2color(depth_img, save_dir):
    depth_img = cv2.imread(depth_img)
    depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=2), cv2.COLORMAP_JET)    # JET RAINBOW
    # depth_color = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)    # JET RAINBOW
    # depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    depth_color = Image.fromarray(depth_color)
    depth_color.save(save_dir)

def generate_heatmap(attention_img, save_dir):
    attention_img = cv2.imread(attention_img)
    heat_map = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)    # JET
    heat_map = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    cv2.imsave(save_dir)


if __name__=='__main__':
    if not os.path.exists(args.color_dir):
        os.mkdir(args.color_dir)
    for img in os.listdir(args.depth_dir):
        print('writing image ', img)
        depth_img = os.path.join(args.depth_dir, img)
        # print(depth_img)
        depth2color(depth_img, os.path.join(args.color_dir, img))

