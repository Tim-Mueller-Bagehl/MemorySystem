"""
    The Interaction Manager controles the entire Systembehavior.

    There are multiple modes of comunication offered to you. 
    Either call the start function and let the System handle the rest.
    Or customise your comunication using the handleManualInput function
"""
from MemorySystem.src.SpeakerIdentification import SpeakerIdentificationSystem
from MemorySystem.src.VectorDatabase import VectorDatabaseSystem
from RealtimeSTT import AudioToTextRecorder
from faster_whisper import WhisperModel
from MemorySystem.src.ModelGateway import ModelGateway
from scipy.io.wavfile import write
from datetime import datetime
from pathlib import Path
import numpy as np
import os




class InteractionManager:



    def __init__(self,*,
                 chatGPTApiKey : str = None,
                 model : str = "gpt-4o-mini",
                 chatGptTemperature : float = 0.7,
                 prompt_for_factextraction : str = None,
                 prompt_for_general_APIcall_retrivedFacts : str = None,
                 prompt_for_general_APIcall_noretrivedFacts : str = None,
                 prompt_to_update_general_Userinformation : str = None,
                 directoryManagementSystemLocation : str | Path = None,
                 voiceRecognitionSystemLocation : str | Path = None,
                 voiceRecognitionMajorityVote : bool = False,
                 inputDeviceIndex : int = None ,
                 language : str = "",
                 Retrieval_similarityIndex : float = None,
                 transcriptionmodel : str = "large-v2",
                 handleResponse = None,
                 general_embeddingstrategy = "bi-encoder_msmarco_bert-base_german",
                 addIdentificationEmbeddings : bool = True,
                 UserDefinedembeddingDimension : int = None
                 ):
        """
        Args:
            chatGPTApiKey (str, optional): your chatGPTapiKey. The System uses the Enviormentvariable OPENAI_API_KEY if no key is provided. Defaults to None.
            model (str, optional): gpt modell. Defaults to "gpt-4o-mini".
            chatGptTemperature (float, optional): LLM temperature. Defaults to 0.7.
            prompt_for_factextraction (str, optional): Customisable prompt for factextraction. The system uses - as a seperator for Facts so that has to be a part of the prompt. Defaults to None.
            prompt_for_general_APIcall_retrivedFacts (str, optional): Customisable Prompt for Responsegeneration, when Information was retrieved. Defaults to None.
            prompt_for_general_APIcall_noretrivedFacts (str, optional): Customisable Prompt for Responsegeneration, when no Information was retrieved. Defaults to None.
            prompt_to_update_general_Userinformation (str, optional): Customisable Prompt to update Userinformation. Defaults to None.
            directoryManagementSystemLocation (str | Path, optional): Location, where the Vectordatabase will live, default is the Packagefolder. Defaults to None.
            voiceRecognitionSystemLocation (str | Path, optional): Location where the Speakeridentification will live, default is the Packagefolder. Defaults to None.
            voiceRecognitionMajorityVote (bool, optional): Defines the Speakeridentifications behavior. Enabeling this is only recomended in a closed off System with no external actors. Defaults to False.
            inputDeviceIndex (int, optional): Device Index of the Microfone (using Pyaudio). Defaults to None.
            language (str, optional): Language of the Speaker, leaving this open probably leads to better results overall. Defaults to "".
            Retrieval_similarityIndex (float, optional): Threshhold for Factretrieval. Defaults to None.
            transcriptionmodel (str, optional): Whisper transcriptionmodell. Defaults to "large-v2".
            handleResponse (_type_, optional): Function to handle the response. Will recieve a String after each Systemresponse. Defaults to None.
            general_embeddingstrategy (str, optional): The System has four supported Encoders. You are free to use any that are supported by SentenceTransformer. Defaults to "bi-encoder_msmarco_bert-base_german".
            addIdentificationEmbeddings (bool, optional): Embeddings generated for Identification will be added to the Vectordatabase afterwards. Defaults to True.
            UserDefinedembeddingDimension (int, optional): Embeddingdimensions have to be provided, when the User chooses to use an Encoder that is not supported. Defaults to None.
        """
        chatGPTApiKey = chatGPTApiKey if chatGPTApiKey else os.getenv("OPENAI_API_KEY")
        match(general_embeddingstrategy):
            case "multi-qa-MiniLM-L6-dot-v1":
                threshold = 0.792
                embeddingDimensions = 384
            case "paraphrase-multilingual-mpnet-base-v2":
                embeddingDimensions = 768
                threshold=0.0
            case "bi-encoder_msmarco_bert-base_german":
                threshold=0.916
                embeddingDimensions = 768
            case "multi-qa-mpnet-base-dot-v1":
                threshold = 0.52
                embeddingDimensions = 768
            case _:
                if embeddingDimensions is None: raise ValueError("Embeddingdimensions have to be provided when an unsupported Encoder is used!")
                embeddingDimensions = UserDefinedembeddingDimension

        if Retrieval_similarityIndex is None: Retrieval_similarityIndex = threshold
        self.VectorDatabase = VectorDatabaseSystem(directoryManagementSystemLocation,embeddingDimensions)
        self.SpeakerIdentification = SpeakerIdentificationSystem(voiceRecognitionSystemLocation,voiceRecognitionMajorityVote,add=addIdentificationEmbeddings)
        self.transcriptionModel = transcriptionmodel
        self.ModelGateway = ModelGateway(self.VectorDatabase,
                                                             apiKey=chatGPTApiKey,model=model,
                                                             temperature=chatGptTemperature,
                                                             similarityIndex=Retrieval_similarityIndex,
                                                             prompt_for_factextraction=prompt_for_factextraction,
                                                             prompt_for_general_APIcall_noretrivedFacts=prompt_for_general_APIcall_noretrivedFacts,
                                                             prompt_for_general_APIcall_retrivedFacts=prompt_for_general_APIcall_retrivedFacts,
                                                             embeddingstrategy=general_embeddingstrategy,
                                                             prompt_to_update_general_Userinformation=prompt_to_update_general_Userinformation
                                                             )
        self.inputDeviceIndex = inputDeviceIndex
        self.language = language
        self.active = False
        self.handleResponse = handleResponse if handleResponse else self.text
        self.shorttermmemory = []
        
    
    

    def transcribe(self,path: str) -> str:
        model_size = self.transcriptionModel
        model = WhisperModel(model_size, device="auto", compute_type="int8")
        segments, info = model.transcribe(
            path,
            vad_filter=True,      
        )
        text = "".join(segment.text for segment in segments)
        return text

    def text(self,response):
        print(response)

    def addNewPerson(self,audioFile : str):
        ID = self.SpeakerIdentification.registerPerson(audioFile)
        transcription = self.transcribe(audioFile)
        self.VectorDatabase.createDirectory(ID)
        self.ModelGateway.retriveAndSaveFacts(ID,transcription,"")
        return ID
    
    def start(self):
        recorder = AudioToTextRecorder(model= self.transcriptionModel,language=self.language,input_device_index=self.inputDeviceIndex)
        self.active = True
        i = 0
        response = ""
        while self.active:
            text = recorder.text()
            print("\n transcription:",text,"\n")
            data = (recorder.audio*32767).astype(np.int16)
            filepath = f"voiceidentification{i}.wav"
            sr = recorder.sample_rate
            write(filepath,sr,data)
            ID  = self.SpeakerIdentification.manageSpeakerIdentification(filepath)
            self.writeProtocol(f"User:{text}")
            if ID:
                memory = self.handleShortTermMemory(ID)
                response = self.ModelGateway.processInput(ID,text,memory.list)
                self.writeProtocol(f"API:{response}")
                memory.addNewEntry(f"User:{text}",f"AI Agent:{response}")
                if i >= 5:
                    i=0
                else:
                    i+=1
                self.handleResponse(response)
            else:
                print("Please register yourself with the system")

        
        self.active = False
        recorder.shutdown()

    def handleManualInput(self, ID :str = None ,audiofile : str = None, transcript : str = None):
        """Second way to comunicate with the System, if raw Microfone Inputs are not given. \n
            The valid Parametercombinations are:\n
            ID,transcript \n
            ID,audiofile \n
            audiofile \n
            audiofile,transcript 

        Args:
            ID (str, optional): ID of the User previously generated by the system. Defaults to None.
            audiofile (str, optional): path to the Audiofile. Defaults to None.
            transcript (str, optional): transcript of the Audiofile. Defaults to None.
        """
        match(ID,audiofile,transcript):

            case(str(id),None,str(text)):
                memory = self.handleShortTermMemory(id)
                response = self.ModelGateway.processInput(id,text,memory)
                self.handleResponse(response)

            case(str(id),str(filepath),None):
                text = self.transcribe(filepath)
                memory = self.handleShortTermMemory(id)
                response = self.ModelGateway.processInput(id,text,memory)
                self.handleResponse(response)

            case(None,str(filepath),None):
                id = self.SpeakerIdentification.manageSpeakerIdentification(filepath)
                if id:
                    text = self.transcribe(filepath)
                    memory = self.handleShortTermMemory(id)
                    response = self.ModelGateway.processInput(id,text,memory)
                    self.handleResponse(response)
                else:
                    print("Please register yourself with the system")

            case(None,str(filepath),str(text)):
                id = self.SpeakerIdentification.manageSpeakerIdentification(filepath)
                if id:
                    memory = self.handleShortTermMemory(id)
                    response = self.ModelGateway.processInput(id,text,memory)
                    self.handleResponse(response)
                else:
                    print("Please register yourself with the system")
            case(_,_,_):
                raise ValueError("Invalid Inputparameters")

    

    def handleShortTermMemory(self,ID):
        """Keeps a Protocoll of the Running Conversation per User. Users that haven't interacted with the System for more than 30 Minutes are automatically removed

        Args:
            ID (_type_): User ID

        Returns:
            _type_: Conversation Protocoll
        """
        now = datetime.now()
        self.shorttermmemory = [a for a in self.shorttermmemory if  ((now - a.timestamp).total_seconds()//60) <=  30]
        memory = next((a for a in self.shorttermmemory if a.id == ID),None)
        if memory is None:
            memory = PersonalMemory(ID)
            self.shorttermmemory.append(memory)
       
        
        return memory

    def stop(self):
        self.active = False

    
    def writeProtocol(self,text : str):
        with open("protocol.txt","a",encoding="utf-8") as f:
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d %H:%M:%S")
            f.write(f"\n{text}      {timestamp}")





class PersonalMemory:

    def __init__(self,id):
        self.id = id
        self.timestamp = datetime.now()
        self.list = []
    
    def addNewEntry(self,question,response):
        self.timestamp = datetime.now()
        self.list.extend([question,response])
        self.managefifo()
    
    def managefifo(self):
        while len(self.list)>20:
            del self.list[:2]