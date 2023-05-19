import numpy as np
import os
import librosa
import h5py


DATASET_PATH = './Dataset_V2_official_classes'
hdf5_path = "data_for_mobNET.h5"

SAMPLE_RATE = 22050
DURATION = 2 #seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION 

n_mels = 128
hop_length = 512
n_fft = 2048

# data augmentation parameters
time_stretch_rates=[0.8, 1.0, 1.2]
pitch_shift_steps=[-2, -1, 0, 1, 2]

# performing tiime stretching and pitch shifting - Data Augmentation
def augment_data(signal, sr, time_stretch_rates, pitch_shift_steps):
  augmented_data = []
  for ts_rate in time_stretch_rates:
      ts_signal = librosa.effects.time_stretch(signal, rate=ts_rate)
      for ps_step in pitch_shift_steps:
          ps_signal = librosa.effects.pitch_shift(ts_signal, sr=sr, n_steps=ps_step)
          augmented_data.append(ps_signal)
  return augmented_data

def save_logMel_dataset(dataset_path, n_mels=n_mels, n_fft=2048, hop_length=hop_length, num_segments=2, overlap=0.5):
  # structure for saving data
  data = {
      "mapping": [],
      "logMelSpec": [],
      "labels": []
  }

  samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
  frame_length = samples_per_segment
  frame_shift = int(frame_length * overlap)

  # cycle through the folders divided by goats feeling
  for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

    if dirpath is not dataset_path:
        # save the mapping names
        dirpath_components = dirpath.split("/") 
        semantic_label = dirpath_components[-1]
        data["mapping"].append(semantic_label)
        print("\nProcessing {}".format(semantic_label))

        # process files for a specific goats feelings folder
        for f in filenames:
            print(f)
            #load audio file
            file_path = os.path.join(dirpath, f)
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            #augment data
            augmented_data = augment_data(signal, sr, time_stretch_rates, pitch_shift_steps)

            #frame the signal and process segments by extracting log mel spectrogram and save data
            for aug_signal in augmented_data:
                frames = librosa.util.frame(aug_signal, frame_length=frame_length, hop_length=frame_shift).T
                for frame in frames:
                  S = librosa.feature.melspectrogram(y=frame, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft) 
                  log_S = librosa.power_to_db(S, ref=np.max)

                  data["logMelSpec"].append(log_S.tolist())
                  data["labels"].append(i-1)

  # Save data in HDF5 format
  with h5py.File(hdf5_path, 'w') as f:
    f.create_dataset("mapping", data=np.array(data["mapping"], dtype='S'))
    f.create_dataset("logMelSpec", data=np.array(data["logMelSpec"]))
    f.create_dataset("labels", data=np.array(data["labels"]))


if __name__ == "__main__":
    save_logMel_dataset(DATASET_PATH, num_segments=2, overlap=0.5)

    
    
    
