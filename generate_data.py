"""
This is an example how to implement real time processing of the DTLN ONNX
model in python.

Please change the name of the .wav file at line 49 before running the sript.
For the ONNX runtime call: $ pip install onnxruntime
    
    

Author: Nils L. Westhausen (nils.westhausen@uol.de)
Version: 03.07.2020

This code is licensed under the terms of the MIT-license.
"""

import soundfile as sf
import numpy as np
import time
import onnxruntime
import librosa
import os
import tarfile as tf
import tqdm


##########################
# the values are fixed, if you need other values, you have to retrain.
# The sampling rate of 16k is also fix.
block_len = 512
block_shift = 128
# load models
interpreter_1 = onnxruntime.InferenceSession('./modified_model_1.onnx', providers=["CPUExecutionProvider"])
model_input_names_1 = [inp.name for inp in interpreter_1.get_inputs()]
# preallocate input
model_inputs_1 = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                dtype=np.float32)
            for inp in interpreter_1.get_inputs()}
# load models
interpreter_2 = onnxruntime.InferenceSession('./model_2.onnx', providers=["CPUExecutionProvider"])
model_input_names_2 = [inp.name for inp in interpreter_2.get_inputs()]
# preallocate input
model_inputs_2 = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                dtype=np.float32)
            for inp in interpreter_2.get_inputs()}

# load audio file
audio,fs = sf.read('noisy.wav')
# check for sampling rate
if fs != 16000:
    audio = librosa.resample(audio, orig_sr=fs, target_sr=16000)
    fs = 16000

# preallocate output audio
out_file = np.zeros((len(audio)))
# create buffer
in_buffer = np.zeros((block_len)).astype('float32')
out_buffer = np.zeros((block_len)).astype('float32')
# calculate number of blocks
num_blocks = (audio.shape[0] - (block_len-block_shift)) // block_shift
# iterate over the number of blcoks  
time_array = []      

model_1_state = np.zeros((1, 2, 128, 2), dtype=np.float32)
model_2_state = np.zeros((1, 2, 128, 2), dtype=np.float32)

os.makedirs("calibration_dataset", exist_ok=True)
input_2_path = os.path.join("calibration_dataset", "input_2")
input_3_path = os.path.join("calibration_dataset", "input_3")
input_4_path = os.path.join("calibration_dataset", "input_4")
input_5_path = os.path.join("calibration_dataset", "input_5")
os.makedirs(input_2_path, exist_ok=True)
os.makedirs(input_3_path, exist_ok=True)
os.makedirs(input_4_path, exist_ok=True)
os.makedirs(input_5_path, exist_ok=True)
tf_input_2 = tf.open(os.path.join("calibration_dataset", "input_2.tar.gz"), "w:gz")
tf_input_3 = tf.open(os.path.join("calibration_dataset", "input_3.tar.gz"), "w:gz")
tf_input_4 = tf.open(os.path.join("calibration_dataset", "input_4.tar.gz"), "w:gz")
tf_input_5 = tf.open(os.path.join("calibration_dataset", "input_5.tar.gz"), "w:gz")

for idx in tqdm.trange(num_blocks):
    start_time = time.time()
    # shift values and write to buffer
    in_buffer[:-block_shift] = in_buffer[block_shift:]
    in_buffer[-block_shift:] = audio[idx*block_shift:(idx*block_shift)+block_shift]
    # calculate fft of input block
    in_block_fft = np.fft.rfft(in_buffer)
    in_mag = np.abs(in_block_fft)
    in_phase = np.angle(in_block_fft)
    # reshape magnitude to input dimensions
    in_mag = np.reshape(in_mag, (1,1,-1)).astype('float32')

    npy_name = f"{idx}.npy"
    np.save(os.path.join(input_2_path, npy_name), in_mag)
    np.save(os.path.join(input_3_path, npy_name), model_1_state)
    tf_input_2.add(os.path.join(input_2_path, npy_name))
    tf_input_3.add(os.path.join(input_3_path, npy_name))

    # set block to input
    model_inputs_1[model_input_names_1[0]] = in_mag
    # run calculation 
    model_outputs_1 = interpreter_1.run(None, model_inputs_1)
    # get the output of the first block
    out_mask, model_1_state = model_outputs_1
    # set out states back to input
    model_inputs_1[model_input_names_1[1]] = model_outputs_1[1]  
    # calculate the ifft
    estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
    estimated_block = np.fft.irfft(estimated_complex)
    # reshape the time domain block
    estimated_block = np.reshape(estimated_block, (1,1,-1)).astype('float32')
    # set tensors to the second block
    # interpreter_2.set_tensor(input_details_1[1]['index'], states_2)
    model_inputs_2[model_input_names_2[0]] = estimated_block

    np.save(os.path.join(input_4_path, npy_name), estimated_block)
    np.save(os.path.join(input_5_path, npy_name), model_2_state)
    tf_input_4.add(os.path.join(input_4_path, npy_name))
    tf_input_5.add(os.path.join(input_5_path, npy_name))

    # run calculation
    model_outputs_2 = interpreter_2.run(None, model_inputs_2)
    # get output
    out_block, model_2_state = model_outputs_2
    # set out states back to input
    model_inputs_2[model_input_names_2[1]] = model_outputs_2[1]
    # shift values and write to buffer
    out_buffer[:-block_shift] = out_buffer[block_shift:]
    out_buffer[-block_shift:] = np.zeros((block_shift))
    out_buffer  += np.squeeze(out_block)
    # write block to output file
    out_file[idx*block_shift:(idx*block_shift)+block_shift] = out_buffer[:block_shift]
    time_array.append(time.time()-start_time)

tf_input_2.close() 
tf_input_3.close()  
tf_input_4.close()  
tf_input_5.close()     