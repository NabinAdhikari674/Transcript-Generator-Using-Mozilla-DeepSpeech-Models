This is a Transcript Generator using Mozilla DeepSpeech pre-trained Models.
The Deepspeech version is 0.9.1. The Docs for it can be found here : https://deepspeech.readthedocs.io/

The pre-trained models can be found and be downloaded here : 
https://deepspeech.readthedocs.io/en/v0.9.1/USING.html#getting-the-pre-trained-model 
OR, https://github.com/mozilla/DeepSpeech/releases

Both .pbmm Model and .scorer language Model are needed to run for this Package.
The Folder for the models to be placed is models/ (Download both .pbmm and .scorer and place it under models folder)

The Audio to be transcriped are to be placed under "audio" folder. All the audio files under the "audio"
folder are transcribed and their relative output transcriptions will be generated on "outputTranscripts" with the same name as the audio file.

The original (truth/real) transcriptions are to be placed under "transcripts" folder while Evaluating and Generating WER/MER/WIL.
The Results/Scores will be generated by name RESULTS.txt.
>>WER = Word Error Rate
>>MER = Match Error Rate
>>WIL = Word Information Lost

Install the all the dependencies on the requirements.txt file.
It can be done by : pip install -r requirements.txt
>> NOTE : For the Package "jiwer" from requirements.txt, Microsoft Visual C++ 14.0 or greater is required.Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/

Apart from dependencies from requirements.txt, you need to install SoX (Sound eXchange) if your Audio file
is not in the recommended 16 KHz Sample Rate for the deepspeech model 
(to convert your audio to 16KHz Sample Rate which is done automatically by a function  if SoX is installed).
To Install SoX :  
Download Sox from : https://sourceforge.net/projects/sox/
And Add Sox your Environment Path Variable.

Finally the working of this Package =>
 
>> Run main.py to transcribe audio files in "audio" folder. The inferred transcript will be generated in "outputTranscripts" folder.

>> Run wordErrorRate.py to get WER,MER,WIL for inferred transcripts from "outputTranscripts" & original(real/true) transcripts from "transcripts" folders.
The results/scores will be stored in RESULTS.txt.
NOTE : the original transcripts will have to have the same name as the inferred transcripts.