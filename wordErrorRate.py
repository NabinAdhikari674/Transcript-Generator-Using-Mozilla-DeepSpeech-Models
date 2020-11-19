#from jiwer import wer
import os
import jiwer
base_directory = os.getcwd()
h_directory = os.path.join(base_directory, "outputTranscripts") # hypothesis/prediction
t_directory = os.path.join(base_directory, "transcripts") # ground truth

transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.RemoveWhiteSpace(replace_by_space=False),
    jiwer.SentencesToListOfWords(word_delimiter=" "),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemovePunctuation()
]) 
txt_file_name = os.path.join(base_directory,"RESULTS.txt")
if os.path.exists(txt_file_name):
    os.remove(txt_file_name)
file_handle = open(txt_file_name, "a+")

for i,path in enumerate (os.listdir(h_directory)):
    next_file = os.path.join(h_directory,path)
    if os.path.isfile(next_file):
        afile = os.path.basename(next_file)
        print("\nProcessing WER For : ",afile)
        with open(next_file, 'r') as file:
            hdata = file.read()
        with open(os.path.join(t_directory,afile), 'r') as file:
            tdata = file.read()
        wer = jiwer.wer(tdata,hdata,
                truth_transform=transformation,
                hypothesis_transform=transformation) # Word Error Rate
        mer = jiwer.wer(tdata,hdata,
                truth_transform=transformation,
                hypothesis_transform=transformation) # Match Error Rate
        wil = jiwer.wer(tdata,hdata,
                truth_transform=transformation,
                hypothesis_transform=transformation) # Word Information Lost
 

        file_handle.write("\n Transcript : "+ afile +" \n\t\t"+
            "   WER : "+ str(wer) +
            "   MER : "+ str(mer) +
            "   WIL : "+ str(wil) +
             "\n")
file_handle.close()

print("## Finished Evaluating Transcripts\n\tResults Stored on : ",txt_file_name)