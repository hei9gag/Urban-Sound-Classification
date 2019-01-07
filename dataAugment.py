import librosa
import os

log_tag= '[Data Augmentation]'
data_set_folder = 'ToiletSoundSet/dataset'
out_dir_prefix = 'ToiletSoundSet/source'
import_folder_paths = []
class_dict = {
    'shower': 0,
    'toilet_flush': 1
}

def _createDirectoryIfNotExist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def _split_sound(in_dir, duration = 3.0):
    folder_name = in_dir.split('/')[-1]
    class_label = class_dict.get(folder_name)
    if class_label is None:
        _log('missing class label for: {}'.format(folder_name))
        return
    out_dir = out_dir_prefix + '/' + folder_name
    _createDirectoryIfNotExist(out_dir)
    counter = 0
    for root, _, files in os.walk(in_dir):
        for file in files:
            _, file_extension = os.path.splitext(file)
            if file_extension != '.wav': continue
            wav_file_path = os.path.join(root, file)

            # load wav
            y, sr = librosa.load(wav_file_path)
            wav_duration = librosa.get_duration(y=y, sr=sr)

            split_times = int(wav_duration / duration)
            offset = 0

            # split the wav into duration
            for _ in range(split_times):
                wav, rate = librosa.load(wav_file_path, offset=offset, duration=duration)
                wav_duration = librosa.get_duration(y=wav, sr=rate)
                if wav_duration < duration:
                    continue

                if counter % 100 == 0:
                    _log('imported sounds:{}'.format(counter))

                file_name = folder_name + '-' + str(counter).zfill(4) + '-' + str(class_label) + '.wav'
                librosa.output.write_wav(out_dir + '/' + file_name, wav, rate)
                offset += duration
                counter += 1
    print('Total sound number for {} is {}'.format(folder_name, str(counter)))

def _import_sounds():
    global import_folder_paths
    for root, dirnames, _ in os.walk(data_set_folder):
        for dir_name in dirnames:
            wav_folder_path = os.path.join(root, dir_name)
            import_folder_paths.append(wav_folder_path)

    for folder_path in import_folder_paths:
        _log('start import sound from folder path:{}'.format(folder_path))
        _split_sound(folder_path)

def _log(message):
    log_message = log_tag + ' ' + message
    print(log_message)

if __name__ == '__main__':
    _import_sounds()