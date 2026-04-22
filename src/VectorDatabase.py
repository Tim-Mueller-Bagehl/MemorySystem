from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import numpy as np
import shutil
import faiss
import json




class VectorDatabaseSystem:

    def __init__(self, root : str | Path = None, embeddingDimension : int = 384):
        if type(root) == str:
            self.DirectoryName = Path(root) / "Memorysystem"
        elif root is None:
            self.DirectoryName = Path(__file__).parent /"MemorySystem"
        else:
            self.DirectoryName = root / "MemorySystem"
        self.embeddingDimension = embeddingDimension
        Path(self.DirectoryName).mkdir(exist_ok=True)
    
        


    def shutDownMemorySystem(self)-> bool:
        """Deletes the Folder Memorysystem where all the userdata and Memorys are stored
        Returns:
            bool: Update on Success
        """
        if Path.exists(self.DirectoryName):
            shutil.rmtree(self.DirectoryName)
            return True
        return False


    def createDirectory(self,ID:str)->bool:
        """Creates a Directory, where a persons Memoryindex, their Photos for Identification and Metadata is saved

        Args:
            ID (str): The persons automaticly generated ID
        """
        Path(self.DirectoryName / ID).mkdir(exist_ok=True)
        index=faiss.IndexFlatIP(self.embeddingDimension)
        FaissFile=self.DirectoryName / ID / (ID+".index")
        faiss.write_index(index,str(FaissFile))
        now=datetime.now()
        now_str=now.strftime("%d-%m-%Y")
        data= {
            "metadata": {
                "lastCleanup" : now_str,
            },
            "chat_history" : [],
            "just_text" : [],
            "general_information" : ""
        }
        with open(self.DirectoryName / ID / "user_data.json", 'w') as f:
            json.dump(data,f,indent=4,ensure_ascii=False)


    def deleteDirectory(self,ID:str)->bool:
        """Deletes a Persons Folder. Only works on valid IDs

        Args:
            ID (str): The Persons ID

        Returns:
            bool: Succes of Failure
        """
        if Path.exists(self.DirectoryName /ID) and self.isID(ID) :
            shutil.rmtree(self.DirectoryName / ID)
            return True
        return False


    def searchMemoryDirectory(self,ID : str, query: np.array, simmilarityIndex : float=0.8, k : int = 5,normalize : bool = False)->List[str]:
        """Searches a persons Memoryindex for simmilar Vectors

        Args:
            ID (str): The persons ID
            query (np.array): The List of Vectors to search the Index
            simmilarityIndex (float, optional): The simmilarity Index decides how simmilar Vectors have to be. A lower Value returns more but less simmilar Vectors. Defaults to 0.8.
            k (int, optional): Amount of simmilar Vectors returned by Faiss during the Process.Adjusting this Value potentially yields more Simmilar Vectors but increases Runtime. Defaults to 5.

        Returns:
            List[str] : The Text of the found vectors
        """
        if normalize: query=self.normalizeVectorList(query)
        index : faiss.Index = faiss.read_index(str(self.DirectoryName / ID / (ID+".index")))
        D : np.ndarray
        I : np.ndarray
        D , I = index.search(query,k)
        idsForReconstruction= I[D >= simmilarityIndex]#Filter all the IDs where D is >= then the similarityindex(threshhold) This gives us the IDs of all the Vectors with related contend
        idsForReconstruction=list(set(idsForReconstruction))    
        recreatedText , generalInformation = self.updateUsedVectorDates(ID,idsForReconstruction)
        
        return recreatedText , generalInformation

    def updateGeneralInformation(self,ID : str, generalInformation : str):
        data = self.loadJson(ID)
        data["general_information"] = generalInformation
        self.saveJson(ID,data)

    def addNewVectorsToDirectory(self,ID : str, listofVectors: np.ndarray, vectorContents : List[str]=None, normalize : bool = False):
        """Adds a List of new Vector to a persons Memory Index the Vectors are Normalised before 
        so calculating the cosine Simmilarity breaks down to the dot Product.

        Args:
            ID (str): The Persons ID
            listofVectors (np.ndarray): List of Vectors to be added to the Index
            vectorContents (List[str],optional) : Users can save saved Information in clear Text in a json File if they choose to. The Information has to be provided in the correct order where listofVectors[0]=VectorContents[0]works. Defaults to None
        """
        if vectorContents is None: vectorContents = []
        if normalize: listofVectors=self.normalizeVectorList(listofVectors)
        if bool(vectorContents):
            duplicates = self.saveChatHistory(ID,vectorContents)
        else:
            self.saveJustDate(ID,listofVectors.shape[0])
            duplicates = []

        if duplicates != []: listofVectors = np.delete(listofVectors,duplicates,axis=0)

        index :faiss.Index = faiss.read_index(str(self.DirectoryName / ID / (ID+".index")))
        index.add(listofVectors)
        faiss.write_index(index,str(self.DirectoryName / ID / (ID+".index")))
        
        
        
        self.cleanUpIndex(ID)
        
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



    def saveChatHistory(self,Folder_name : str , listOfText : List[str]):
        """Saves the information saved in the Vector aswell as the ID of the Vector and the date when it was saved

        Args:
            text (str): Text before it was embedded
        """
        filteredDuplicates=[]
        data = self.loadJson(Folder_name)

        now=datetime.now()
        now_str=now.strftime("%d-%m-%Y")
        ID= data["chat_history"][-1][0]+1 if data["chat_history"] else 0

        for index,text in enumerate(listOfText):

            if text in data["just_text"]:
                #Filters duplicates
                filteredDuplicates.append(index)
                continue

            new_message= [ID,now_str,text]
            data["chat_history"].append(new_message)
            data["just_text"].append(text)
            ID+=1
        self.saveJson(Folder_name,data)
        
        return filteredDuplicates

    def saveJustDate(self,Folder_name , numberOfVectors : int):
        """If you don't want to save clear text just save the Date when the Vector was saved and it's ID

        Args:
            Folder_name (str) : Name of the Folder where the Json userdata is saved
            numberOfVectors (int): The number of Vectors that have been saved
        """
        data = self.loadJson(Folder_name)

        now=datetime.now()
        now_str=now.strftime("%d-%m-%Y")
        ID= data["chat_history"][-1][0]+1 if data["chat_history"] else 0
        listOfElements=[[ ID + i ,now_str,""] for i in range(numberOfVectors)]
        data["chat_history"].append(listOfElements)

        self.saveJson(ID,data)

    def cleanUpIndex(self,ID : str , timeframe : int = 6):
        """Removes all Vectors that haven't been accessed in 6 Months from the Index and the Json File

        Args:
            ID (str): ID of the Index
            timeframe (int, optional): Timeframe of the Cleanup in Months. Defaults to 6.
        """
        if timeframe <= 0: return 

        data = self.loadJson(ID)

        currentDate = datetime.now()
        oneWeekAgo = currentDate - timedelta(days=7)
        lastCleanup=datetime.strptime(data["metadata"]["lastCleanup"],"%d-%m-%Y")
        if lastCleanup < oneWeekAgo: #We want to clean up at least once a week
            return
        data["metadata"]["lastCleanup"] = datetime.strftime(currentDate,"%d-%m-%Y")

        sixMonthAgo = currentDate - timedelta(timeframe*30)
        oldEntrys=[]

        for chat in data["chat_history"]:
            chatDateStr=chat[1]
            chatDate=datetime.strptime(chatDateStr,"%d-%m-%Y")
            if chatDate < sixMonthAgo:
                oldEntrys.append(chat)
        
        data["chat_history"] = [entry for entry in data["chat_history"] if entry not in oldEntrys or entry == data["chat_history"][-1]]#We need to preserve the last entry to keep track of the max id
        if bool(oldEntrys):
            index :faiss.Index = faiss.read_index(str(self.DirectoryName / ID / (ID+".index")))
            index.remove_ids(oldEntrys[1])
            faiss.write_index(index,str(self.DirectoryName / ID / (ID+".index")))
        
        self.saveJson(ID,data)
        
    def updateUsedVectorDates(self,ID: str, ListofVectorIDs : List[int]):
        """Updates the date in the user_data.json for every Vector in the List to the current day

        Args:
            ID (str): ID of the Index
            ListofVectorIDs (List[int]): List of VectorIDs that get their date updated
        """
        data = self.loadJson(ID)
        now = datetime.now()
        now_str=now.strftime("%d-%m-%Y")

        recreatedText = []
        for entry in data["chat_history"]:
            if entry[0] in ListofVectorIDs:
                entry[1]=now_str
                recreatedText.append(entry[2])
        
        self.saveJson(ID,data)
        return recreatedText , data["general_information"]
        
        

    def isID(self,ID:str)->bool:
        try:
            ID=int(ID)
        except ValueError:
            print("This is not a Valid ID. IDs Consist of Numbers")
            return False
        return ID > 0


    def loadJson(self,ID : str):
        if not self.isID(ID):
            return 
        try:
            with open(self.DirectoryName / ID / "user_data.json", 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print("JSON file is corrupted. Starting fresh")
            data={"metadata" : {}, "chat_history" : [],"just_text" : [],"general_information" : ""}
        except Exception as e:
            print(f"Unexpected error: {e}")
            data={"metadata" : {}, "chat_history" : [],"just_text" : [],"general_information" : ""}
            
        return data
    
    def saveJson(self, ID : str , data):
        if not self.isID(ID):
            return
        with open(self.DirectoryName / ID / "user_data.json", 'w') as f:
            json.dump(data,f,indent=4,ensure_ascii=False) 

