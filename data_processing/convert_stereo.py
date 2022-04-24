################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Geoff Pascoe (gmp@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################

import argparse
import os
import re
import matplotlib.pyplot as plt
from datetime import datetime as dt
from image import load_image
from camera_model import CameraModel
import cv2
import time

parser = argparse.ArgumentParser(description='Play back images from a given directory')

parser.add_argument('--dir', type=str, default=r"..\large_data\stereo\right", help='Directory containing images.')
parser.add_argument('--output_dir', type=str, default=r"..\large_data\stereo_processed\right", help='Output directory to save the demosaiced and undistorted images')

parser.add_argument('--models_dir', type=str, default="../models", help='(optional) Directory containing camera model. If supplied, images will be undistorted before display')
parser.add_argument('--scale', type=float, default=1.0, help='(optional) factor by which to scale images before display')

args = parser.parse_args()

camera = re.search('(stereo|mono_(left|right|rear))', args.dir).group(0)

timestamps_path = os.path.join(os.path.join(args.dir, os.pardir, camera + '.timestamps'))

if not os.path.isfile(timestamps_path):
  timestamps_path = os.path.join(args.dir, os.pardir, os.pardir, camera + '.timestamps')
  if not os.path.isfile(timestamps_path):
      raise IOError("Could not find timestamps file")

model = None
if args.models_dir:
    model = CameraModel(args.models_dir, args.dir)

current_chunk = 0
timestamps_file = open(timestamps_path)
os.makedirs(args.output_dir, exist_ok=True)

for line in timestamps_file:
    tokens = line.split()
    datetime = dt.utcfromtimestamp(int(tokens[0])/1000000)
    chunk = int(tokens[1])

    filename = os.path.join(args.dir, tokens[0] + '.png')
    if not os.path.isfile(filename):
        if chunk != current_chunk:
            print("Chunk " + str(chunk) + " not found")
            current_chunk = chunk
        continue

    current_chunk = chunk
    start_time = time.time()
    img = cv2.imread(filename)
    # print(img == model.undistort(img))
    img = load_image(filename, model)
    print(time.time()-start_time)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out_filename = os.path.join(args.output_dir, tokens[0] + '.png')
    cv2.imwrite(out_filename, img)

