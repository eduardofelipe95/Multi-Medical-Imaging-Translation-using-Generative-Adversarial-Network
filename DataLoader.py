
import scipy
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from PIL import Image
print("imports ok")
class DataLoader():
    def __init__(self, img_res, pathImageTrainAPaired, pathImageTrainAUnpaired, typeA, typeB, typeC, imageRegion="full", seed=0):
        self.img_res = img_res
        self.pathImageTrainAPaired = pathImageTrainAPaired
        self.pathImageTrainAUnpaired = pathImageTrainAUnpaired
        self.typeA = typeA
        self.typeB = typeB
        self.typeC = typeC
        self.imageRegion = imageRegion
        self.patientsForTrainTestPaired = []
        self.patientsForTrainTestUnpaired = []


        np.random.seed(seed)
        
    def datasetDescription(self):
        patients = sorted(glob(self.pathImageTrainAPaired + "/*"))
        count = 0
        print("Descricao do dataset de imagens pareadas em %s" % self.typeA)
        for i, patient in enumerate(patients):
            name = (patient.split("/"))[-1]
            numberOfImages = len(glob(patient + "/*"))
            print("%d| Paciente %s tem %d imagens" % (i + 1, name, numberOfImages))
            count += numberOfImages

        print("Temos um total de %d imagens" % count)
    
    def holdOut(self, nPatientsTrain, randomSample = False):
        patients = sorted(glob(self.pathImageTrainAPaired + "/*"))
        unpairedPatients = sorted(glob(self.pathImageTrainAUnpaired + "/*"))
        countTrainA, countTrainB, countTrainC = 0, 0, 0
        countTestA, countTestB, countTestC = 0, 0, 0

        if randomSample:
            patients = np.random.choice(patients, len(patients), replace=False)

        patientsForTraining, patientsForTesting = [], [] 
        for i, patient in enumerate(patients):
            name = (patient.split("/"))[-1]

            if int(nPatientsTrain) > i:
                patientsForTraining.append(name)
                countTrainA += len(glob(patient + "/*" ))
                countTrainB += len(glob(patient.replace("/%s/" % self.typeA , "/%s/" % self.typeB ) + "/*"))
                countTrainC += len(glob(patient.replace("/%s/" % self.typeA , "/%s/" % self.typeC ) + "/*"))      
            else:
                patientsForTesting.append(name)
                countTestA += len(glob(patient + "/*"))
                countTestB += len(glob(patient.replace("/%s/" % self.typeA , "/%s/" % self.typeB ) + "/*" ))
                countTestC += len(glob(patient.replace("/%s/" % self.typeA , "/%s/" % self.typeC ) + "/*" ))
                
        
        print("Pareadas| Temos um total de %d imagens de %s, %d de %s e %d de %s para treinamento, os pacientes %s foram selecionados" % (countTrainA, self.typeA, countTrainB, self.typeB, countTrainC , self.typeC, patientsForTraining))
        print()
        print("Pareadas| Temos um total de %d imagens de %s, %d de %s e %d de %s para teste, os pacients %s foram selecionados" % (countTestA, self.typeA, countTestB, self.typeB, countTestC, self.typeC, patientsForTesting))
        print()
        
        countTrainA, countTrainB, countTrainC = 0, 0, 0
        
        patientsUnpairedForTraining = []
        
        for i, patient in enumerate(unpairedPatients):
            name = (patient.split("/"))[-1]
            
            if int(nPatientsTrain) > i:
                #print(name)
                countTrainA += len(glob(patient + "/*"))
                countTrainB += len(glob(patient.replace("/%s/" % self.typeA , "/%s/" % self.typeB ) + "/*" ))
                countTrainC += len(glob(patient.replace("/%s/" % self.typeA , "/%s/" % self.typeC ) + "/*"))
                patientsUnpairedForTraining.append(name)
        
        
        #patientsUnpairedForTraining = sorted(list(set(patientsUnpairedForTraining).symmetric_difference(patientsForTesting)))
        
        
        
        
        print("Nao Pareadas| Temos um total de %d imagens de %s, %d de %s e %d de %s para treinamento, os pacientes %s foram selecionados" % (countTrainA, self.typeA, countTrainB, self.typeB, countTrainC , self.typeC, patientsUnpairedForTraining))
        print()
        
        print("Nao Pareadas| Temos um total de %d imagens de %s, %d de %s e %d de %s para teste, os pacients %s foram selecionados" % (countTestA, self.typeA, countTestB, self.typeB, countTestC, self.typeC, patientsForTesting))
        print()
        
        
        
        self.patientsForTrainTestPaired =  [patientsForTraining, patientsForTesting]
        self.patientsForTrainTestUnpaired = [patientsUnpairedForTraining, patientsForTesting]
    def kFold(self, k):
        patients = sorted(glob(self.pathImageTrainAPaired + "/*"))

        lis = []

        patientsForTraining, patientsForTesting = [], [] 
        foldSize = int(len(patients)/k) 
        for i in range(k):
            lis.append(patients[:foldSize])
            patients = patients[foldSize:]
        if len(patients) > 0:
            lis.append(patients)
        trainTest = []

        for i in range(len(lis)):
            trainlist = []
            testList = []
            testList.append(lis[i])

            for j in range(len(lis) - 1):
                if i != j:
                    trainlist += lis[j]

            trainTest.append([trainlist, testList])

        self.patientsForTrainTestPaired =  trainTest

    def leaveOneOut(self):
        patients = sorted(glob(self.pathImageTrainAPaired + "/*"))
        k = len(patients)
        lis = []
        patientsForTraining, patientsForTesting = [], [] 
        foldSize = int(len(patients)/k) 
        for i in range(k):
            lis.append(patients[:foldSize])
            patients = patients[foldSize:]
        if len(patients) > 0:
            lis.append(patients)

        trainTest = []

        for i in range(len(lis)):
            trainlist = []
            testList = []
            testList.append(lis[i])

            for j in range(len(lis) - 1):
                if i != j:
                    trainlist += lis[j]

            trainTest.append([trainlist, testList])

        self.patientsForTrainTestPaired =  trainTest


    
    def loadBatchTest(self, batch_size):
        
        pathsA = []
        for patient in self.patientsForTrainTestPaired[1]:
            pathsA += sorted(glob(self.pathImageTrainAPaired + "/%s/*" % (patient)))
        
        pathsA = np.random.choice(pathsA, len(pathsA), replace=False)
        batch_images = np.random.choice(pathsA, size=batch_size)
        
        imgs_A = []
        imgs_B = []
        imgs_C = []
        
        for img_path in batch_images:
            img_A = Image.open(img_path)
            img_B = Image.open(img_path.replace("/%s/" % self.typeA, "/%s/" %  self.typeB))
            
            quadrant = (0,0,self.img_res[0], self.img_res[0])
            img_A = img_A.crop(quadrant)
            img_B = img_B.crop(quadrant)
            
            img_A = img_A.resize((self.img_res[0],self.img_res[1]))
            img_B = img_B.resize((self.img_res[0],self.img_res[1]))
            
            img_A = np.array(img_A)
            img_B = np.array(img_B)
            
            img_A = img_A.reshape(self.img_res[0],self.img_res[1],self.img_res[2])
            img_B = img_B.reshape(self.img_res[0],self.img_res[1],self.img_res[2])

            imgs_A.append(img_A)
            imgs_B.append(img_B)
            
            try:
                img_C = Image.open(img_path.replace("/%s/" % self.typeA, "/%s/" %  self.typeC))
                img_C = img_C.crop(quadrant)
                img_C = img_C.resize((self.img_res[0],self.img_res[1]))
                img_C = np.array(img_C)
                img_C = img_C.reshape(self.img_res[0],self.img_res[1],self.img_res[2])
                imgs_C.append(img_C)
            except FileNotFoundError:
                pass
            

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.
        
        if len(imgs_C) > 0:
            imgs_C = np.array(imgs_C)/127.5 - 1.
        
        return imgs_A, imgs_B, imgs_C
        
    
    def loadFullTest(self):
        
        pathsA = []
        
        for patient in self.patientsForTrainTestPaired[1]:
            pathsA += sorted(glob(self.pathImageTrainAPaired + "/%s/*" % (patient)))
            #print(self.pathImageTrainAPaired + "/%s/*" % (patient))
        
        #print(len(pathsA))
                
        imgs_A = []
        imgs_B = []
        imgs_C = []
        
        
        for img_path in pathsA:
            img_A = Image.open(img_path)
            img_B = Image.open(img_path.replace("/%s/" % self.typeA, "/%s/" %  self.typeB))

            quadrant = (0,0,self.img_res[0], self.img_res[0])
            img_A = img_A.crop(quadrant)
            img_B = img_B.crop(quadrant)
            
            img_A = img_A.resize((self.img_res[0],self.img_res[1]))
            img_B = img_B.resize((self.img_res[0],self.img_res[1]))
            
            img_A = np.array(img_A)
            img_B = np.array(img_B)
            
            img_A = img_A.reshape(self.img_res[0],self.img_res[1],self.img_res[2])
            img_B = img_B.reshape(self.img_res[0],self.img_res[1],self.img_res[2])

            imgs_A.append(img_A)
            imgs_B.append(img_B)
            
            try:
                img_C = Image.open(img_path.replace("/%s/" % self.typeA, "/%s/" %  self.typeC))
                img_C = img_C.crop(quadrant)
                img_C = img_C.resize((self.img_res[0],self.img_res[1]))
                img_C = np.array(img_C)
                img_C = img_C.reshape(self.img_res[0],self.img_res[1],self.img_res[2])
                imgs_C.append(img_C)
            except FileNotFoundError:
                pass
        
        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.
        if len(imgs_C) > 0:
            imgs_C = np.array(imgs_C)/127.5 - 1.
        
        print("Pareadas| Temos um total de %d imagens de %s, %d de %s e %s %s para testes" % (len(imgs_A), self.typeA,len(imgs_B), self.typeB, len(imgs_C), self.typeC))
        return imgs_A, imgs_B, imgs_C

    def loadBatchPaired(self, batch_size=1):
        
        path = []
        for patient in self.patientsForTrainTestPaired[0]:
            path += sorted(glob(self.pathImageTrainAPaired + "/%s/*" % (patient)))
        
        path = np.random.choice(path, len(path), replace=False)
        self.n_batches = int(len(path) / batch_size)
        
        for i in range(self.n_batches):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B, imgs_C  = [], [], []
            for img in batch:
                img_A = Image.open(img)
                img_B = Image.open(img.replace("/%s/" % self.typeA, "/%s/" %  self.typeB))
                
                quadrant = (0,0,self.img_res[0], self.img_res[0])
                img_A = img_A.crop(quadrant)
                img_B = img_B.crop(quadrant)

                img_A = img_A.resize((self.img_res[0],self.img_res[1]))
                img_B = img_B.resize((self.img_res[0],self.img_res[1]))
                
                img_A = np.array(img_A)
                img_B = np.array(img_B)
                
                
                img_A = img_A.reshape(self.img_res[0], self.img_res[1], self.img_res[2])
                img_B = img_B.reshape(self.img_res[0], self.img_res[1], self.img_res[2])

                imgs_A.append(img_A)
                imgs_B.append(img_B)
                
                try:
                    img_C = Image.open(img.replace("/%s/" % self.typeA, "/%s/" %  self.typeC))
                    img_C = img_C.crop(quadrant)
                    img_C = img_C.resize((self.img_res[0],self.img_res[1]))
                    img_C = np.array(img_C)
                    img_C = img_C.reshape(self.img_res[0], self.img_res[1], self.img_res[2])
                    imgs_C.append(img_C)
                except FileNotFoundError:
                    pass


            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.
            if len(imgs_C) > 0:
                imgs_C = np.array(imgs_C)/127.5 - 1.
                
            yield imgs_A, imgs_B, imgs_C
            
    def loadBatchUnpaired(self, batch_size=1):
        
        path_A, path_B, path_C = [], [], []
        for patient in self.patientsForTrainTestPaired[0]:
            path = []
            path_A += sorted(glob(self.pathImageTrainAUnpaired + "/%s/*" % (patient)))
            
            path_B += sorted(glob(self.pathImageTrainAUnpaired.replace("/%s" % self.typeA,
                                                               "/%s" %  self.typeB) + "/%s/*" % (patient)))
            path_C += sorted(glob(self.pathImageTrainAUnpaired.replace("/%s" % self.typeA,
                                                               "/%s" %  self.typeC) + "/%s/*" % (patient))) 
        
        
        path_A = np.random.choice(path_A, len(path_A), replace=False)
        path_B = np.random.choice(path_B, len(path_B), replace=False)       
        path_C = np.random.choice(path_C, len(path_C), replace=False)
        

        self.n_batches = int(min(len(path_A), len(path_B), len(path_C)) / batch_size)
        
        total_samples = self.n_batches * batch_size

        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)
        path_C = np.random.choice(path_C, total_samples, replace=False)

        for i in range(self.n_batches):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            batch_C = path_C[i*batch_size:(i+1)*batch_size]
            
            imgs_A, imgs_B, imgs_C = [], [], []
            for img_A, img_B, img_C  in zip(batch_A, batch_B, batch_C):
                img_A = Image.open(img_A)
                img_B = Image.open(img_B)  

                quadrant = (0,0,self.img_res[0]/2, self.img_res[0]/2)
                img_A = img_A.crop(quadrant)
                img_B = img_B.crop(quadrant)
                
                img_A = img_A.resize((self.img_res[0],self.img_res[1]))
                img_B = img_B.resize((self.img_res[0],self.img_res[1]))
    
                img_A = np.array(img_A)
                img_B = np.array(img_B)   
                
                img_A = img_A.reshape(self.img_res[0],self.img_res[1],self.img_res[2])
                img_B = img_B.reshape(self.img_res[0],self.img_res[1],self.img_res[2])

                imgs_A.append(img_A)
                imgs_B.append(img_B)
                try:
                    img_C = Image.open(img_C)
                    img_C = img_C.crop(quadrant)
                    img_C = img_C.resize((self.img_res[0],self.img_res[1]))
                    img_C = np.array(img_C)
                    img_C = img_C.reshape(self.img_res[0],self.img_res[1],self.img_res[2])
                    imgs_C.append(img_C)
                except FileNotFoundError:
                    pass


            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.
            
            if len(imgs_C) > 0:
                imgs_C = np.array(imgs_C)/127.5 - 1.
            
            yield imgs_A, imgs_B, imgs_C
