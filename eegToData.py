# -*- coding: utf-8 -*-
from config import slicesPath

import trackdata

import png
import numpy as np
import collections

def int_val(f):
    v = int(5*f)
    v = min(v, 255)
    v = max(v, 0)
    return v

def fft_color(sig):
    vals = np.empty(len(sig))
    for i in range(len(sig)):
        vals[i] = int_val(sig[i])
    return vals

def fft(sig):
    win = np.hanning(len(sig))
    samples = np.array(sig, dtype='float64')
    samples *= win
    f = abs(np.fft.rfft(samples, norm='ortho'))
    f = np.delete(f, 0)
    return f

def dump_image(arr, subfolder, image_name):
    png.from_array(arr, 'L', info={ "bitdepth" : 8 }).save("{}/{}/{}.png".format(slicesPath, subfolder, image_name))

enjoy_to_class = {
	-2 : "very_bad",
	-1 : "bad",
	0 : "ok",
	1 : "good",
	2 : "very_good"
}

def generate_slices(session_id, channel, sample_rate=256, max_image_len=128, window=90):
    dbconn = trackdata.DBConnection(session_id, 'eeg')
    increment = int(sample_rate*(1-window/100))

    samples = collections.deque(maxlen=sample_rate)

    shift = 0
    png_arr = np.zeros([max_image_len, max_image_len])
    img_idx = 0
    enjoy = 0
    png_arr_idx = 0
    for x in dbconn.doc:
        if "channel_data" in x:
            U = x["channel_data"][channel]
            if len(samples) == samples.maxlen:
                shift += 1
            samples.append(U)
            if shift == increment:
                shift = 0
                f = fft(samples)
                png_arr[png_arr_idx] = fft_color(f)
                png_arr_idx += 1
            if png_arr_idx == max_image_len:
                dump_image(png_arr, enjoy_to_class[enjoy], '{}_{}_ch{}'.format(session_id, img_idx, channel))
                png_arr_idx = 0
                img_idx += 1
        elif "event_name" in x:
            if x["event_name"] == "enjoy_changed":
                enjoy = x["value"]