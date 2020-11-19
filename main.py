import argparse
import os
import sys
import shutil
from tqdm import tqdm
from deepspeech import Model, version
from segmentAudio import silenceRemoval
from audioProcessing import extract_audio, convert_samplerate
#from writeToFile import write_to_file
import re
import numpy as np
import wave

def sort_alphanumeric(data):
    """Sort function to sort os.listdir() alphanumerically
    Helps to process audio files sequentially after splitting 

    Args:
        data : file name
    """
    
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    
    return sorted(data, key = alphanum_key)
    
def ds_process_audio(ds, audio_file, file_handle):  
    """Run DeepSpeech inference on each audio file generated after silenceRemoval
    and write to file pointed by file_handle

    Args:
        ds : DeepSpeech Model
        audio_file : audio file
        file_handle : txt file handle
    """

    fin = wave.open(audio_file, 'rb')
    fs_orig = fin.getframerate()
    desired_sample_rate = ds.sampleRate()
    # Check if sampling rate is required rate (16000)
    # won't be carried out as FFmpeg already converts to 16kHz
    if fs_orig != desired_sample_rate:
        print("Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition".format(fs_orig, desired_sample_rate), file=sys.stderr)
        audio = convert_samplerate(audio_file, desired_sample_rate)
    else:
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
    fin.close()

    # Perform inference on audio segment
    infered_text = ds.stt(audio)
    #print(">>> Limits : ",limits)
    if len(infered_text) != 0:
        file_handle.write(infered_text + "\n")

def main():
    print("\nTranscript Generator Using Mozilla DeepSpeech\n")

    parser = argparse.ArgumentParser(description="Transcript Generator Using Mozilla DeepSpeech\n")
    parser.add_argument('--folder',required=False,
                        help='Input Audio Folder')
    
    args = parser.parse_args()

    ds_model = 'models/deepspeech-0.9.1-models.pbmm'
    ds_scorer = 'models/deepspeech-0.9.1-models.scorer'

    ds = Model(ds_model)
    ds.enableExternalScorer(ds_scorer)

    base_directory = os.getcwd()
    output_directory = os.path.join(base_directory, "outputTranscripts")
    if args.folder :
        audio_directory = os.path(args.folder)
    else :
        audio_directory = os.path.join(base_directory, "audio")
    buffer_dir = "audioBuffer"
    
    print("\n\tAudio Folder Path is : ",audio_directory)

    for i,path in enumerate (os.listdir(audio_directory)):
        next_file = os.path.join(audio_directory,path)
        if os.path.isfile(next_file):
            afile = next_file.split('\\')[-1]
            print("\nProcessing For : ",afile)
            txt_file_name = os.path.join(output_directory,afile[0:-4]+".txt")
            if os.path.exists(txt_file_name):
                os.remove(txt_file_name)
            file_handle = open(txt_file_name, "a+")
            print("\tSplitting the AudioFile in parts on Silent Parts of the Audio File ... ",end=" ")
            working_dir = os.path.join(audio_directory,buffer_dir)
            if not os.path.exists(working_dir):
                os.mkdir(working_dir)
            #print("\n\tWorking dir : ",working_dir)
            silenceRemoval(buffer_dir,next_file)
            print("Done.\n\tRunning Inference ... ")
            for file in tqdm(sort_alphanumeric(os.listdir(working_dir))):
                audio_segment_path = os.path.join(working_dir, file) 
                ds_process_audio(ds, audio_segment_path, file_handle)
            file_handle.close()
            print("\n ## Inference Finished for ",afile," ##")
            shutil.rmtree(working_dir)
            print("\n\t TXT file Saved to", txt_file_name,"\n")
    
            

if __name__ == "__main__":
    main()
