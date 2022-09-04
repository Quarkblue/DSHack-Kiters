import pyttsx3

def speak(labelId):

    engine = pyttsx3.init(); #AUDIO ENGINE (PUT "sapi5" IN BRACKETS WHEN PORTED TO WINDOWS)

    engine.setProperty('rate',140)
    voices = engine.getProperty('voices')

    engine.setProperty('voice',voices[2].id) #67 62 56 

    LABELIDDICT = {
        1:"Hello",
        2:"Thanks",
        3:"Yes",
        4:"No",
        5:"I Love You",
        6:"Sleep",
        7:"Stop",
        8:"Sad",
        9:"Play",
        10:"Play"
    }

    engine.say(LABELIDDICT[labelId]);
    engine.runAndWait();
