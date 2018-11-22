# -*- coding: utf-8 -*-
from config import slicesPath

import trackdata

import png
import numpy as np
import collections

def int_val(f):
    v = int(10*f)
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
    -2 : "bad",
	-1 : "bad",
	0 : "ok",
	1 : "good",
	2 : "good"
}

def generate_slices(session_id, channel, max_image_len, window):
    print("Creating slices for session {}, channel {}, image_size {}, window {}".format(session_id, channel, max_image_len, window))
    sample_rate=max_image_len*2
    dbconn = trackdata.DBConnection()
    doc = dbconn.session_data(session_id, 'eeg')
    increment = int(sample_rate*(1-window/100))

    samples = collections.deque(maxlen=sample_rate)

    shift = 0
    png_arr = np.zeros([max_image_len, max_image_len])
    img_idx = 0
    enjoy = 0
    png_arr_idx = 0
    for x in doc:
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
    
    print("Created {} images".format(img_idx))

def generate_slices_all(channel, max_image_len, window):
    dbconn = trackdata.DBConnection()
    for session_id in dbconn.all_sessions():
        generate_slices(session_id, channel, max_image_len, window)