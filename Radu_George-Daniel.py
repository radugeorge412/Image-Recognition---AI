#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import pandas as pd
import numpy as np
import os


# In[5]:


from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pdb
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


# In[6]:


date_workout = pd.read_csv("train.txt", header=None, names=["cale", "eticheta"])


# In[42]:


workout_nume = date_workout.loc[:, "cale"].tolist()
workout_etichete = date_workout.loc[:, "eticheta"].tolist()
workout_nume = [os.path.join('train', i) for i in workout_nume]


# In[8]:


date_test = pd.read_csv("test.txt", header=None, names=["cale", "eticheta"])


# In[9]:


test_nume = date_test.loc[:, "cale"].tolist()
test_etichete = date_test.loc[:, "eticheta"].tolist()
test_nume = [os.path.join('test', j) for j in test_nume]


# In[10]:


date_validation = pd.read_csv("validation.txt", header=None, names=["nume", "eticheta"])


# In[11]:


validation_nume = date_validation.loc[:, "nume"].tolist()
validation_etichete = date_validation.loc[:, "eticheta"].tolist()
validation_nume = [os.path.join('validation', k) for k in validation_nume]


# In[13]:



class set_workout(Dataset):
    def __init__(self, workout_nume, workout_etichete):
        self.workout_nume = workout_nume
        self.workout_etichete = workout_etichete
        
    def __len__(self):
        return len(self.workout_etichete)

    def __getitem__(self, index):
        workout_path = self.workout_nume[index]
        img = Image.open(workout_path)
        img = np.array(img) / 255
        imagine_intoarsa = torch.tensor(img).unsqueeze(0).float()
        eticheta = torch.tensor(self.workout_etichete[index])
        
        return imagine_intoarsa, eticheta


# In[14]:


class set_test(Dataset):
    def __init__(self, test_nume, test_etichete):
        self.test_nume = test_nume
        self.test_etichete = test_etichete
        
    def __len__(self):
        return len(self.test_nume)

    def __getitem__(self, index):
        test_path = self.test_nume[index]
        img = Image.open(test_path)
        img = np.array(img) / 255
        imagine_intoarsa = torch.tensor(img).unsqueeze(0).float()
        eticheta = torch.tensor(self.test_etichete[index])
        
        return imagine_intoarsa, eticheta


# In[15]:


class set_validation(Dataset):
    def __init__(self, validation_nume, validation_etichete):
        self.validation_nume = validation_nume
        self.validation_etichete = validation_etichete
        
    def __len__(self):
        return len(self.validation_nume)

    def __getitem__(self, index):
        validation_path = self.validation_nume[index]
        img = Image.open(validation_path)
        img = np.array(img) / 255
        imagine_intoarsa = torch.tensor(img).unsqueeze(0).float()
        eticheta = torch.tensor(self.validation_etichete[index])
        
        return imagine_intoarsa, eticheta


# In[43]:


tr = set_workout(workout_nume, workout_etichete)
data_loader_workout = DataLoader(tr, batch_size=32, shuffle=True)
  
tt = set_test(test_nume, test_etichete)
data_loader_test = DataLoader(tt, batch_size=1, shuffle=False)

tv= set_validation(validation_nume, validation_etichete)
data_loader_validation = DataLoader(tv, batch_size=32, shuffle=False)


# In[18]:


import torch.nn as nn
import torch.nn.functional as F
class RaduG(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=5, stride = 1)
        self.conv2 = nn.Conv2d(in_channels=40, out_channels=120, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=120, out_channels=80, kernel_size=5, stride=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(320, 70)
        self.fc2 = nn.Linear(70, 3)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x


# In[19]:


import torch.optim as optim
model = RaduG()
cr = nn.CrossEntropyLoss()


# In[44]:


optimizer = optim.Adam(model.parameters(), lr=0.00001)


# In[1]:



model.train()
for e in range(10):
    for item in data_loader_workout:
        img, label = item

        optimizer.zero_grad()
        iesire = model(img)

        loss = cr(iesire, label)

        loss.backward()
        optimizer.step()
        print(loss)
        print("Epoca: "+ str(e))
        
    


# In[46]:


lista_predictii=[]
model.eval()
for item in data_loader_validation:
    img, label = item
    iesire = model(img)
    lista_predictii = np.concatenate((lista_predictii, torch.argmax(iesire, axis=1).cpu().numpy()))
acuratete = accuracy_score(lista_predictii,validation_etichete)
print(str(acuratete))
  


# In[40]:


lista=[]

test_data_nou = np.loadtxt("test.txt", dtype=str)
with open("submissionTEST.csv", "w") as g:  
    g.write("id,label\n")
    with torch.no_grad():
        for item in data_loader_test:
            images, labels = item
            iesire = model(images)
            predicted = torch.argmax(iesire,1)
            #print(predicted)
            
            var = np.array(predicted)   
            lista.append(var)
            
                
lista= np.array(lista)

output = zip(test_data_nou, lista)
with open("submissionTEST.csv", "w") as g:  
    g.write("id,label\n")
    for element in output:
        g.write(str(element[0] + "," + str(element[1][0])) + "\n")


# In[28]:





# In[ ]:




