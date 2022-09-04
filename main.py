from concurrent.futures import thread
from pty import master_open
from textToSpeech import speak
from asyncio import wrap_future
from tkinter import *
from threading import *
from time import sleep
from PIL import Image   
from cmath import rect
from turtle import color

# main window frame
root = Tk();
root.geometry("560x350")
root.title('My ASL Speaker');
root.configure(background="#A5F1E9")
bg = PhotoImage(file="GUI/hand.png")
ico = PhotoImage(file="GUI/icon.png")
root.iconphoto(False,ico)
bgLabel = Label(root,image = bg);
bgLabel.place(x = 0 , y = 1 , relwidth=1,relheight=1);


class Gui(thread):
    master = 0
    def __init__(self, master):
        self.master = master
        devFrame = Frame(master)
        devFrame.pack()

        # #title of the app
        titleLabel = Label(master, text= "WELCOME TO MY ASL SPEAKER",font=('Comic Sans MS', 26), bg= "#A5F1E9",fg="#EB1D36").pack()

        #brief note
        descLabel = Label(master, text="Click Start Now to begin understanding what your specially abled friends have to say",font=('Montserrat',10), bg= "#0F3460",fg ="white").pack()


        #start button and positioning
        sbFrame = LabelFrame(master)
        sbFrame.pack(side= BOTTOM);
        startButton = Button(sbFrame, text="Start Now!",padx = 50, command = self.translate,font=('Comic Sans MS',15),bg = "#FFEEAF",fg="#16213E")
        startButton.pack()


        self.startButton = Button(master,text = "Start Now", command = self.translate)

    def translate(self):
        for widget in self.master.winfo_children():
            widget.destroy()

        gameCanvas = Canvas(self.master,bg= "red")
        gameCanvas.pack();
        self.drawExit();


    def drawExit(self):
        ebFrame = LabelFrame(self.master)
        ebFrame.pack(side=BOTTOM)
        exitButton = Button(self.master,text = "Exit Program" , padx=50 , command=self.master.quit)
        exitButton.pack(side= BOTTOM);





e = Gui(root);

root.mainloop();