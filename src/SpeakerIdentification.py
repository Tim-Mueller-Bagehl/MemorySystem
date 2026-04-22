from speechbrain.inference.speaker import SpeakerRecognition
from scipy.io.wavfile import read,write
from collections import Counter
from pydub import AudioSegment
from pathlib import Path
import noisereduce as nr
import soundfile as sf
import numpy as np
import librosa
import random
import shutil
import faiss
import torch
import json
import time
import os 




device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",savedir="pretrained_models/spkrec-ecapa-voxceleb")
class SpeakerIdentificationSystem:

    def __init__(self,root : str | Path = None, marjorityVote : bool = False,autoremove : bool = False, add : bool = True):
        if type(root) == str:
            root = Path(root)
        elif root is None:
            root = Path(__file__).parent / "VoiceRecognition"
        self.DirectoryName = root
        self.Index = root /"Voicerecognition.index"
        self.Json = root / "Voicerecognition.json"
        self.TempDirectory = root / "Tmp"
        self.embeddingDimension = 192
        self.majorityVote = marjorityVote
        self.autoremove = autoremove
        self.add = add
        self.initVoicerecognition()




    def initVoicerecognition(self):
        """Creates a Folder with the Name VoiceRecognition. 
        The Voiceembeddings are saved in an Index in set folder aswell as a Json that relates User-IDs with Embedding-IDs for Identification
        """
        Path(self.DirectoryName).mkdir(exist_ok=True)
        Path(self.TempDirectory).mkdir(exist_ok=True)
        if not self.Index.exists():
            index=faiss.IndexFlatIP(self.embeddingDimension)
            faiss.write_index(index,str(self.Index))
            data= {
                "User_info" : [] # Userinfo[i] = ID where i is the ID of the embedding
            }
            self._saveJson(data)

    def shutDownVoicerecognition(self):
        """Deletes the Folder Voicerecognition including all the Embeddings and the Userdata

        Returns:
            bool: True on Success
        """
        if Path.exists(self.DirectoryName):
            shutil.rmtree(self.DirectoryName)
            return True
        return False

    def clearTempDirectory(self):
        """Removes every File stored in Temp that are older than 30 Minutes
        """
        Threshhold=time.time()-(30*60)

        for a in self.TempDirectory.iterdir():
            if a.is_file():
                modificationTime= a.stat().st_mtime
                if modificationTime < Threshhold:
                    a.unlink()

    def normalizeVectorList(self,embeddings: np.ndarray)->np.ndarray:
        """Just Normalises a List of Vectors

        Args:
            list (np.ndarray): List of Vectors to be Noramlised

        Returns:
            np.ndarray: The normalised List
        """
        if embeddings.ndim == 1:
            embeddings /= np.linalg.norm(embeddings)
        else:
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
    
    #originale Implementation übernommen von: https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/inference/encoders.html
    def encode_file(self,audiofile : str, normalize : bool = False) -> np.array:
        #I copyed it because there was no version of it in the Speaker recoginition class but the functions to recreate it exist
        waveform = model.load_audio(str(audiofile))
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        result = model.encode_batch(batch,rel_length,normalize=normalize)
        return result.squeeze(1).numpy()

    def convertAudioFileToEmbedding(self,Audiofile : str ) -> np.array:
        """Turns an Audiosample into a new Embedding by splitting the File into 2 sec long parts and creating an Embedding for each Part.
        We take the mean over all the Embeddings after that to get one Embedding for the Person. Should be for People to be saved in the Index

        Args:
            Audiofile (str): Location of the Audiofile. Has to be of Type Wave

        Raises:
            ValueError: Audiofile is not of Type Wave

        Returns:
            np.array: Calculated Embedding
        """

        Audiofile = self.ensureCorrectWavFormat(Audiofile)
        path = self.preProcessing(Audiofile)
        return self.encode_file(path)

            
    def preProcessing(self,AudioFile : str):
        """Filters Backgroundnoise

        Args:
            AudioFile (str): Name of The Audiofile
        """
        AudioFileBaseName = os.path.basename(AudioFile)
        AudioFileBase=os.path.splitext(AudioFileBaseName)[0]
        data, rate = librosa.load(AudioFile, sr=None)
        reduced = nr.reduce_noise(y=data, sr=rate)
        path = self.TempDirectory / (str(AudioFileBase) + "cleaned.wav")
        sf.write(path, reduced, rate)
        if self.autoremove:
            Path(AudioFile).unlink(missing_ok=True)
        return path


    def addEmbedding(self,Embeddings : np.ndarray, UserIDs : str):
        """Adds normalized Embeddings to the Voicerecognition.index and saves the IDs in the Voicerecognition.Json

        Args:
            Embeddings (np.array): Already Calculated Embeddings
            UserIDs (List[str]): IDs of the related People where Embedding[i] -> UserID[i]
        """

        index : faiss.Index = faiss.read_index(str(self.Index))
        Embeddings = self.normalizeVectorList(Embeddings)
        index.add(Embeddings)
        faiss.write_index(index,str(self.Index))
        self.saveUserInformationInJson(UserIDs)

    def saveUserInformationInJson(self,User : str):
        """Adds Data entry for every added Embedding in the Format: [Embedding-ID , User-ID]

        Args:
            UserInformation (List[str]): List of User IDs wich had Embeddings added to the Index
        """
        data=self._getJson()
        
        data["User_info"].append(User)
        
        self._saveJson(data)

    def searchIndex(self,Embedding : np.array, k : int=5, simmilarityIndex : float=0.25)->str:
        """Takes an Embedding as Input and searches for simmilar Samples. It finds the most likely 
        Person through a Votinprocess. Where the person with the most matching Vectors counts as Identified

        Args:
            Embedding (np.array): The Embedding of the Voicemessage
            k (int, optional): Amount of Vectors returned by index.search(). Defaults to 5.
            simmilarityIndex (float, optional): Cosine Simmilarity Boundry. Defaults to 0.25. copyed from Speechbrain:https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/inference/speaker.html#SpeakerRecognition

        Returns:
            str: ID of the Identified person
        """
        Embedding = self.normalizeVectorList(Embedding)
        index : faiss.Index = faiss.read_index(str(self.Index))
        D : np.ndarray
        I : np.ndarray
        D , I = index.search(Embedding,k)

        idsForMatching=I[D >= simmilarityIndex]
        data = self._getJson()
        try:
            matches =[data["User_info"][i] for i in idsForMatching]
        except IndexError: #data is empty
            return None 
        if matches == []:
            return None
        if self.majorityVote:
            ID = Counter(matches).most_common(1)[0][0]
            if self.add:
                index.add(Embedding)
                faiss.write_index(index,str(self.Index))
                self.saveUserInformationInJson(ID)
            return ID
        else:
            if matches.count(matches[0]) != 5:
                return None
            else:
                ID = matches[0]
                if self.add:
                    index.add(Embedding)
                    faiss.write_index(index,str(self.Index))
                    self.saveUserInformationInJson(ID)
                return ID


    def generateID(self,maxID:int=10000):
        """Generates a New ID for a new Person. Also creates a Folder with set ID as the Name and a Memory Index.

        Args:
            maxID (int, optional): Max value for IDs cant be below 100. Defaults to 1000.
        """
        newID="0"
        while True and maxID > 100:
            newID=str(random.randrange(1,maxID))
            if not Path.exists(self.DirectoryName / newID):
                break
        if(newID=="0"):
            print("choose a higher max ID all IDs between 1 and {maxID} are taken")
            return        
        return newID

    def registerPerson(self,audioFile : str):
        """The system works best if we already have at least 3 Embeddings per person. This function splits the input that should be around one minute long into 5 pieces 
        and encodes each so the Voicerecognition is able to recognise the person

        Args:
            audioFile (str): path to the audiofile

        Returns:
            str: ID of the newly registered person
        """
        ID = self.generateID()
        audioFileBase = os.path.splitext(audioFile)[0]
        sr , audio = read(str(audioFile))
        N = audio.shape[0]
        size = N // 5
        start = 0
        for i  in range(5):
            end = start + size
            part = audio[start:end] if i < 4 else audio[start:]
            filename = audioFileBase + f"part{i}.wav"
            write(filename= filename,data=part,rate=sr)
            self.encodeFileAndAddToIndex(filename,ID)
            Path(filename).unlink(missing_ok=True)
        return ID
        

    def encodeFileAndAddToIndex(self,audiofile : str, ID : str):
        embedding = self.convertAudioFileToEmbedding(audiofile)
        self.addEmbedding(embedding,ID)


    def ensureCorrectWavFormat(self,audioFile : str):
        """This function converts the input audifile into a Format that works for the Embedding strategy I use for Voicerecognition.

        Args:
            audioFile (str): Location of the Audiofile

        Returns:
            _type_: Location of the new Audiofile(this doesn't change if the File was of Type .wav)
        """
        path = Path(audioFile)
        info = sf.info(audioFile)
        if info.format == "WAV" and info.subtype in "PCM_16" and info.samplerate == 16000 and info.channels == 1:#Necessary Wav format
            return audioFile
        
        audio = AudioSegment.from_file(path)
        audio = audio.set_frame_rate(16000)
        audio = audio.set_channels(1)
        wavPath = path.with_suffix(".wav")
        audio.export(wavPath,format = "wav",codec="pcm_s16le")
        return wavPath

        

    def _getJson(self) ->dict[str,list]:
        """Loads the Voicerecognition.json File

        Returns:
            dict[str,list]: Voicerecognition.json
        """
        try:
            with open(self.Json, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:#Lets hope this never happends we have to reset the Index on this aswell...
            print("JSON file is corrupted. Starting fresh")
            data={"User_info" : []}
        except Exception as e:
            print(f"Unexpected error: {e}")
            data={"User_info" : []}
        return data

    def _saveJson(self,data : dict[str,list]):
        """Saves the Voicerecognition.json file

        Args:
            data (dict[str,list]): The full File to be saved
        """
        with open(self.Json ,'w') as f:
            json.dump(data,f,indent=4,ensure_ascii=False)




    def manageSpeakerIdentification(self,audioFile : str):
        """Manages the whole SpeakerIdentification Pipeline. A new ID is generated and a new MemoryIndex is created if the Person can't be Identified.
        Returns the ID of the person.


        Args:
            audioFile (): location of the Audiofile

        Returns:
            _type_: ID 
        """
        if not Path(audioFile).exists():
            raise FileNotFoundError()

        embedding = self.convertAudioFileToEmbedding(audioFile)
        ID = self.searchIndex(embedding)
        self.clearTempDirectory()
        return ID 