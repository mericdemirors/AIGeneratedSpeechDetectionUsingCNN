# Creates image dataset from audio files using stingray library.

print("imports...", flush=True)
import os
import argparse

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import wavfile

from stingray.lightcurve import Lightcurve
from stingray.bispectrum import Bispectrum

# audio to image conversion parameters
def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('-path', '-p', type=str, default="", help='directory to save images')
    parser.add_argument('-start', '-s', type=int, default=0, help='index of first processed audio')
    parser.add_argument('-end', '-e', type=int, default=0, help='index of last processed audio')
    parser.add_argument('-segment_size', '-ss', type=int, default=400, help='segment(window) size to slide on audio')
    parser.add_argument('-segment_overlap', '-so', type=int, default=200, help='how much does each segment(window) overlap with eachother')
    return parser.parse_args()

args = parse_arguments()
save_folder_path = args.path
start = args.start
end = args.end
SEGMENT_SIZE = args.segment_size
SEGMENT_OVERLAP = args.segment_overlap

real_audio_df = pd.read_csv("")
fake_audio_df = pd.read_csv("")
real_audio_df = real_audio_df.sort_values(["seconds"]).reset_index(drop=True)
fake_audio_df = fake_audio_df.sort_values(["seconds"]).reset_index(drop=True)

features = ["absolutes","angles","reals","imags", "cum3s"]
for f in features:
    os.makedirs(os.path.join(save_folder_path, f, "real_audio"), exist_ok=True)
    os.makedirs(os.path.join(save_folder_path, f, "fake_audio"), exist_ok=True)

# gets cum3 and RC_layers from audio
def get_features(audio_path, segment_size=SEGMENT_SIZE, overlap=SEGMENT_OVERLAP, max_K=-1):
    samplerate, data = wavfile.read(audio_path)
    num_segments = (len(data) - segment_size) // overlap + 1
    if max_K > 0:
        num_segments = min(num_segments, max_K)

    RC_layers = np.zeros((num_segments, segment_size+1, segment_size+1), dtype=complex)
    cum3_sum = np.zeros((segment_size+1, segment_size+1))

    time_values = np.linspace(0, segment_size / samplerate, segment_size)
    for idx, segment_start in enumerate(range(0, len(data), overlap)):
        if idx == num_segments:
            break
        segment = data[segment_start:segment_start + segment_size]
        
        lc = Lightcurve(time_values, segment)
        bs = Bispectrum(lc, window="hamming")

        mag, phase, cum3 = bs.bispec_mag, bs.bispec_phase, bs.cum3
        cum3_sum = cum3_sum + cum3

        R = mag * np.cos(phase)
        C = mag * np.sin(phase)
        RC_layers[idx] = R + C * 1j

    return RC_layers, cum3_sum/num_segments

# creates the signature image from RC_layers matrix
def create_signature_image(RC_layers):
    RC_layers = RC_layers[..., np.newaxis]
    signature_image = np.zeros(RC_layers.shape[1:], dtype=complex)
    tops = np.sum(RC_layers, axis=0)

    signature_image = np.reshape(np.array([tops[r][c]/(np.sqrt(np.dot(RC_layers[:,r,c,:].T,np.conjugate(RC_layers[:,r,c,:])).real) + 0.0001) 
                                        for r in range(signature_image.shape[0]) 
                                        for c in range(signature_image.shape[1])]), signature_image.shape)

    # list comprehension is for this for loop
    # for r in range(signature_image.shape[0]):
    #     for c in range(signature_image.shape[1]):
    #         L = RC_layers[:,r,c,:]
    #         top = tops[r][c]
    #         bottom = np.sqrt(np.dot(L.T, np.conjugate(L)).real)
    #         signature_image[r,c] = top/(bottom + 0.0001)

    return signature_image

# saves each feature image to corresponding foller
def save_images(signature_image, cum3_avg, absolute_path, angle_path, real_path, imag_path, cum3_path):
    absolute = np.absolute(signature_image)
    absolute_norm = (absolute - absolute.min()) / (absolute.max() - absolute.min())
    cv2.imwrite(absolute_path, (absolute_norm*255).astype(np.uint8))

    angle = np.angle(signature_image)
    angle_norm = (angle - angle.min()) / (angle.max() - angle.min())
    cv2.imwrite(angle_path, (angle_norm*255).astype(np.uint8))

    real = signature_image.real
    real_norm = (real - real.min()) / (real.max() - real.min())
    cv2.imwrite(real_path, (real_norm*255).astype(np.uint8))

    imag = signature_image.imag
    imag_norm = (imag - imag.min()) / (imag.max() - imag.min())
    cv2.imwrite(imag_path, (imag_norm*255).astype(np.uint8))

    cum3_norm = (cum3_avg - cum3_avg.min()) / (cum3_avg.max() - cum3_avg.min())
    cv2.imwrite(cum3_path, (cum3_norm*255).astype(np.uint8))

# create real part of image dataset
print("real images...", flush=True)
for i in tqdm(range(start, end)):
    path = real_audio_df["path"][i]
    image_name = path.split(os.sep)[-1][:-4]

    [absolute_path,angle_path,real_path,imag_path,cum3_path] = [os.path.join(save_folder_path, f, "real_audio" ,image_name + ".png") for f in features]

    if os.path.isfile(absolute_path) and os.path.isfile(angle_path) and os.path.isfile(real_path) and os.path.isfile(imag_path) and os.path.isfile(cum3_path):
        continue

    RC_layers, cum3_avg = get_features(path, max_K=-1)
    signature_image = create_signature_image(RC_layers)
    save_images(signature_image, cum3_avg, absolute_path, angle_path, real_path, imag_path, cum3_path)

# create fake part of image dataset
print("fake images...", flush=True)
for i in tqdm(range(start, end)):
    path = fake_audio_df["path"][i]
    image_name = path.split(os.sep)[-1][:-4]
    [absolute_path,angle_path,real_path,imag_path,cum3_path] = [os.path.join(save_folder_path, f, "fake_audio" ,image_name + ".png") for f in features]

    if os.path.isfile(absolute_path) and os.path.isfile(angle_path) and os.path.isfile(real_path) and os.path.isfile(imag_path) and os.path.isfile(cum3_path):
        continue

    RC_layers, cum3_avg = get_features(path, max_K=-1)
    signature_image = create_signature_image(RC_layers)
    save_images(signature_image, cum3_avg, absolute_path, angle_path, real_path, imag_path, cum3_path)

