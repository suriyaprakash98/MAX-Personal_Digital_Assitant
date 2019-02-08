# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 12:08:22 2018

@author: Baakchsu
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import pyttsx3;
import tensorflow as tf
import numpy as np
import facenet
from align import detect_face
import cv2
import argparse
import urllib.request
import spacy
# =============================================================================
# import tensorflow as tf
# import tensornets as nets
# from tensornets.datasets import coco
# =============================================================================
import requests
from bs4 import BeautifulSoup
import speech_recognition as sr
import pyaudio
def train (data, config_file, model_dir):
     training_data = load_data(data)
     configuration = config.load(config_file)
     trainer = Trainer(configuration)
     trainer.train(training_data)
     model_directory = trainer.persist(model_dir, fixed_model_name = 'chat')
from gtts import gTTS
import pygame
import os
import smtplib
import speech_recognition as sr
import pyaudio
import random
from datetime import datetime

from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
 #from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer, Metadata, Interpreter
from rasa_nlu import config
engine = pyttsx3.init();
# =============================================================================
#     while True:
#         clock.tick(500)
#         if     
# =============================================================================
train('./data/training_data.json', './config/config.yml', './models/nlu')
import numpy as np
import pandas as pd
db=pd.read_csv("./db.csv")
#db=db.loc[:,"known"]

#variables=[timeofday,person,just_started]
#db=[known]
# =============================================================================
# global light_stat
# global presence
# global person
# =============================================================================
light_stat=0
presence=1
person='anirudh'

interpreter = Interpreter.load('./models/nlu/default/chat')
#dic={'greet':'greet','goodbye':'bye','ask_time':'ask_time','tell_day':tell_day(),'tell_health':tell_health(),'ask_boutyou':ask_boutyou(),'complement':complement(),'ask_date':ask_date()}
timesince=int(str(datetime.time(datetime.now()))[:2])
#db=np.load("db.npy")

init = 1
def who_creator():
    var=random.choice(["My creator is the organisation rotronix.","Rotronix is my creator organisation.","Rotronix is my god."])  
    return var          
def out():
    global presence,person,light_stat
    if person in db["known"]:
        
        if timeofday is "morning" or "evening" or "afternoon":
            var=random.choice(["hey!"+timeofday,"hi!"+timeofday,"hello"+timeofday])
        else:
            var=random.choice(["","hey!","hi!","hello"])+person
    else:
       if timeofday is "morning" or "evening" or "afternoon":
         var=random.choice(["Hi there!","hey there!","hi!"+timeofday,"hello"+timeofday])
       else:
           var = random.choice(["Hi there!","hey there!","hi!","hello there!"])    
    return var 
          
def greet():
    global presence,person,light_stat
    var=""
    if person in db["known"]:
        
        if timeofday is "morning" or "evening" or "afternoon":
            var=random.choice(["hey! Good "+timeofday,"hi! Good "+timeofday,"hello! Good "+timeofday])
        else:
            var=random.choice(["","hey! ","hi! ","hello "])+person
    else:
       if timeofday is "morning" or "evening" or "afternoon":
         var=random.choice(["Hi there!","hey there!","hi! Good "+timeofday,"hello! Good "+timeofday])
       else:
           var = random.choice(["Hi there!","hey there!","hi!","hello there!"])
    return var       
def goodbye():
    global presence,person,light_stat
    if timeofday is "night":
       if person in db["known"]:
         var=random.choice(["Good night. Sweet dreams!","See you tomorrow!","Good night. Sleep tight","See you tomorrow, "+person+"!","Good night, "+person+".","Bye! Good night!","Goodbye! See you tomorrow!"])
       else:
           var=random.choice(["Good night","Good night. See you later!","Good night. See you"])
    else:
       if person in db["known"]:
           var=random.choice(["See you then, bye.","Goodbye, "+person,"Bye. see you later "+person,"See you then!","Bye. See you later."])
       else:
            var=random.choice(["See you then, bye.","Goodbye.","Bye. see you later","See you then!"])
        
    return var    
def complement():
    global presence,person,light_stat
    if person in db["known"]:
        var=random.choice(["Thank you so much.","Thanks! I'm glad to be with you","Thank you!","Thank you very much!","I'm glad I could be of help to you","Glad to hear this. Thank you so much!"])
    else:
        var=random.choice(["Thank you so much.","Thank you!","Thank you very much!","I'm glad I could be of help to you","Glad to hear this. Thank you so much!"])
  
    return var

def ask_look():
    global presence,person,light_stat
    if person in db["known"]:
        var=random.choice(["Great as always!","Perfect!","You look fabulous, "+person+"!","You look amazing, "+person+"!","You look great as usual!","You look stunning, "+person,"Exquisite!","Magnificent!","You seem marvellous,"+person+"!","Charming!","Delightful!","Gorgeous!"])
    else:
        var=random.choice(["Great as always!","Perfect!","You look fabulous!","You look amazing!","You look great as usual!","You look stunning","Exquisite!","Magnificent!","You seem marvellous!","Charming!","Delightful!","Gorgeous!"])
  
    return var
    
def ask_time():
    global presence,person,light_stat
    time=str(datetime.time(datetime.now()))[:5]
    if person in db["known"]:
       if timeofday is "morning":
           var="It is "+time+" in the morning, "+person
       else:    
        var=random.choice(["The time is "+time,"It is "+time+", "+person,"Time is "+time,"It's "+time])
    else:
        var=random.choice(["The time is "+time,"It is "+time,"Time is "+time,"It's "+time])
    return var
def ask_who():
    var=random.choice(["hi! I'm Maximus, I am an advanced ai chatbot from Rotronix ","Hey there! I'm Maximus. You can call me Max.","Hi! I'm maximus. Glad to meet you!","Hey there, I'm max.","Hi! I'm Maximus, your personal assistant." ])
    return var
def tell_who():
    var=random.choice(["Hey "+person,'Hi '+person,'Your name is '+person])
    return var

    
def ask_date_day():
    
    date=str(datetime.now().date())
   
    
    if person in db["known"]:
         
        var=random.choice(["The date is "+date,"It is "+date+", "+person,"Date is "+date,"It's "+date])
       
            
    else:
        var=random.choice(["The date is "+date,"It is "+date,"It's "+date])
    return var
# =============================================================================
# def ask_date_day() :
#    
#     
#     var=random.choice([str(datetime.now().date())])
# =============================================================================
    #return var
def tell_health():
    if person in db["known"]:
        var=random.choice(["Try to take some rest, "+person+" .","Please consider taking some rest.",person+", please get some sleep.","Try to get some sleep and rest.","Drink some water and please take rest, "+person+" ."])      
    return var    
# =============================================================================
# def ask_schedule():
#    
#     if person in db["known"]:
#         var=random.choice(["Your schedule for today is "+schedule[person]])
#     elif schedule[person]=="":
#         var=random.choice(["You dont have anything in your schedule for today"])
#     else:
#         var=random.choice(["I don't know your schedule"])
#     return var
# =============================================================================
def comm_on():
    if person in db["known"]:
        
        var=random.choice(["Turning on the lights now!", "Doing it right away.","Done! lights on."])
    else:
        var=random.choice(["Sorry, you don't have authorization","Oops! Sorry, you can't give IoT commands"])
    return var     
def comm_off():
    if person in db["known"]:
        
        var=random.choice(["Turning off the lights now!", "Doing it right away.","Done! lights off."])
    else:
        var=random.choice(["Sorry, you don't have authorization.","Oops! Sorry, you can't give IoT commands."])
    return var     
def ask_boutyou():
    var=random.choice(["Thanks for asking. I'm doing great!","I'm well, thanks.","I'm good, thanks for asking!","I'm doing good. How 'bout you?","I'm doing great. how are you?"])
    return var  
def affirm_ans():
    var=random.choice(["Good to know that you're doing good.","Great!","Awesome! Good to know.","Super! Good to hear!"])
    return var
def send_mail():
    def send_email(subject, msg):
     try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login('jasper72010@gmail.com','demolisa1')
        message = 'Subject: {}\n\n{}'.format(subject, msg)
        server.sendmail('jasper72010@gmail.com','jasper72010@gmail.com', message)
        server.quit()
        print("Success: Email sent!")
     except:
        print("Email failed to send.")


    subject = "Emergency"
    msg = "Please attend to "+person+ " immedietly."

    send_email(subject, msg)
 # from rasa_nlu.converters import load_data
# =============================================================================
# def dialogue_flow(inps=''):
#     
#     count=0
#     
#     mic=0
#     while(1):
#         global presence,person,light_stat
#         if count//2==0 and not mic:
#             
#             presence,person=1,'anirudh'
#         sen=input("\n\n")
#         res=interpreter.parse(sen)
#         
#         try:
#               parsed=exec(res['intent']['name']+'()')
#               print(res['intent']['name']+'()')
#               if res['intent']['name']=='comm':
#                   if 'on' in sen and light_stat==0:
#                       comm_on()
#                       light_stat=1
#                   elif 'off' in sen and light_stat==1:
#                       comm_off()
#                       light_stat=0
#               
#               else:
#                
#                
#                
#               
#                print(parsed,"none")
#                
#         except:
#                print("I couldn't get you. Please come again")
#         if res['intent']['name']=='goodbye':
#             break
#         count += 1
# =============================================================================

while(1):
   if int(str(datetime.time(datetime.now()))[:2])<=10:
        timeofday='morning'
   elif  int(str(datetime.time(datetime.now()))[:2])<=13 and int(str(datetime.time(datetime.now()))[:2])>12:
        timeofday='afternoon'
   elif int(str(datetime.time(datetime.now()))[:2])>15 and int(str(datetime.time(datetime.now()))[:2])<22:
        timeofday='evening'
   elif int(str(datetime.time(datetime.now()))[:2])>=22:
        timeofday='night'
   presence,person=1,'anirudh'            #yolo()  
   lastpresence=int(str(datetime.time(datetime.now()))[:2])
# =============================================================================
#     if presence==1 and abs(timesince-int(str(datetime.time(datetime.now()))[:2]))>=4:
#      timesince=int(str(datetime.time(datetime.now()))[:2])
#      #   out()
#         
# # =============================================================================
# #      dialogue_flow()
# # =============================================================================
#      count=0
#     
#      mic=0
#      while(1):
# # =============================================================================
# #         global presence,person,light_stat
# # =============================================================================
#         if count//2==0 and not mic:
#             
#             presence,person=1,'anirudh'
#         sen=input("\n\n")
#         res=interpreter.parse(sen)
#         
#         try:
#               parsed=exec(res['intent']['name']+'()')
#               print(res['intent']['name']+'()')
#               if res['intent']['name']=='comm':
#                   if 'on' in sen and light_stat==0:
#                       comm_on()
#                       light_stat=1
#                   elif 'off' in sen and light_stat==1:
#                       comm_off()
#                       light_stat=0
#               
#               elif parsed!='goodbye':
#                
#                
#                
#               
#                print(parsed,"none")
#               elif parsed=='goodbye':
#                   break
# 
#         except:
#             print("I didn't get you.")
#     elif init==1:
# =============================================================================
   
   if presence==1: 
    count=0
    #print(eval('greet()'))
    mic=0
    lastpresence=int(str(datetime.time(datetime.now()))[:2])
    while(1):
        if count//2==0 and not mic:
            
            presence,person=1,'anirudh'
        text=''
        r=sr.Recognizer()
        with sr.Microphone() as source:
            
         print('say')
         audio = r.listen(source)
         try:
             text=r.recognize_google(audio)
             print(text)
         except:
             print("Couldn't recognise")
        #sen=input("\n\n")
        res=interpreter.parse(text)
        
        if res['intent']['confidence']>=.20:
            parsed=eval(res['intent']['name']+'()')
        #sen=parsed
        try:
         if res['intent']['name']=='comm' and res['intent']['confidence']>=30:
              if 'on' in text and light_stat==0:
                      
                      light_stat=1
              elif 'off' in text and light_stat==1:
                      
                      light_stat=0
              
         else:
               if res['intent']['confidence']>=.20:
                   #sss=random.randint(1,100909091)
                   engine.say(parsed);
                   engine.runAndWait()
                   
                   #print(parsed,'\n\n',res['intent'])
               else:
                   if len(text)!=0:
                       engine.say("I didn't get you. Please come again.");
                       engine.runAndWait()
                       print("I didn't get you. Please come again.")

               #print("I couldn't get you. Please come again")
         if res['intent']['name']=='goodbye':
            break
         count += 1    
        except:
            print("I didn't get you")
        
# =============================================================================
#         init=0
# =============================================================================
# =============================================================================
#          dialogue_flow()
# =============================================================================
    

         