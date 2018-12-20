import trackdata

import png
import numpy as np
import collections

from freqlogminmax import minmax

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

def fft(sig, norm=None):
    win = np.hanning(len(sig))
    samples = np.array(sig, dtype='float64')
    samples *= win
    f = abs(np.fft.rfft(samples, norm))
    f = np.delete(f, 0)
    return f

def fft_log(sig):
    return np.log(fft(sig))

def fft_elements(sig):
    f = fft_log(sig)
    elements = []
    elements.append(sum(f[:5]))     #delta
    elements.append(sum(f[4:9]))    #theta
    elements.append(sum(f[8:14]))   #alpha
    elements.append(sum(f[13:31]))  #beta
    elements.append(sum(f[30:45]))  #gamma
    elements.append(sum(f[65:]))     #other
    # scaling to 0..1
    #for freq_bin in range(45):
    #    freq = f[freq_bin]
    #    r_min = minmax[freq_bin][0]
    #    r_max = minmax[freq_bin][1]
    #    freq -= r_min
    #    freq /= (r_max - r_min)
    #    freq = max(freq, 0.0)
    #    freq = min(freq, 1.0)
    #    elements.append(freq)
    #elements = np.log(elements)
    return elements

def dump_image_impl(imagesPath, arr, subfolder, image_name):
    png.from_array(arr, 'L', info={ "bitdepth" : 8 }).save("{}/{}/{}.png".format(imagesPath, subfolder, image_name))

def dump_image(imagesPath, arr, subfolder, image_name):
    size = int(arr.shape[0]/2)
    img = np.rot90(arr[:size,:size])
    dump_image_impl(imagesPath, img, subfolder, image_name + '_1')
    img = np.rot90(arr[size:,:size])
    dump_image_impl(imagesPath, img, subfolder, image_name + '_2')

enjoy_to_class = {
    -2 : "bad",
	-1 : "bad",
	0 : "ok",
	1 : "good",
	2 : "good"
}

def generate_slices(imagesPath, session_id, channel, max_image_len, window):
    print("Creating slices for session {}, channel {}, image_size {}, window {}".format(session_id, channel, max_image_len, window))
    max_image_len *= 2
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
                f = fft(samples, norm='ortho')
                png_arr[png_arr_idx] = fft_color(f)
                png_arr_idx += 1
            if png_arr_idx == max_image_len:
                dump_image(imagesPath, png_arr, enjoy_to_class[enjoy], '{}_{}_ch{}'.format(session_id, img_idx, channel))
                png_arr_idx = 0
                img_idx += 1
        elif "event_name" in x:
            if x["event_name"] == "enjoy_changed":
                enjoy = x["value"]
    
    print("Created {} images".format(2*img_idx))

def generate_slices_all(imagesPath, channel, max_image_len, window):
    dbconn = trackdata.DBConnection()
    for session_id in dbconn.all_sessions():
        generate_slices(imagesPath, session_id, channel, max_image_len, window)