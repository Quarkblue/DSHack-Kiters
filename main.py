from textToSpeech import speak
from asyncio import wrap_future
from tkinter import *
from PIL import Image

# main window frame
winFrame = Tk();
winFrame.geometry("560x350")
winFrame.title('My ASL Speaker');
bg = PhotoImage(file="GUI/hand.png")
ico = PhotoImage(file="GUI/icon.png")
winFrame.iconphoto(False,ico)
bgLabel = Label(winFrame,image = bg);
bgLabel.place(x = 0 , y = 1 , relwidth=1,relheight=1);



def translate():
    for widget in winFrame.winfo_children():
        widget.destroy()

    gameCanvas = Canvas(winFrame,bg= "red")
    gameCanvas.pack();
    drawExit();

    # output function comes here


    return 

#fucntion to draw a exit button on the screen
def drawExit():
    ebFrame = LabelFrame(winFrame)
    ebFrame.pack(side=BOTTOM)
    exitButton = Button(winFrame,text = "Exit Program" , padx=50 , command=winFrame.quit)
    exitButton.pack(side= BOTTOM);


#title of the app
titleLabel = Label(winFrame, text= "WELCOME TO MY ASL SPEAKER",font=('Helvetica bold', 26), bg= "white").pack()

#brief note
descLabel = Label(winFrame, text="Click Start Now to begin understanding what your specially abled friends have to say", bg="white").pack()


#start button and positioning
sbFrame = LabelFrame(winFrame)
sbFrame.pack(side= BOTTOM);
startButton = Button(sbFrame, text="Start Now!",padx = 50, command = translate,bg = "white")
startButton.pack()

 

#event loop
winFrame.mainloop();