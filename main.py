from tkinter import *
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import sys
from MLP import *

class PrintConsole(): #class to print everything appears in console in a specific textbox in GUI
    def __init__(self, textbox): 
        self.textbox = textbox 

    def write(self, text):
        self.textbox.insert(tk.END, text)

    def flush(self):
        self.textbox.delete('1.0', END)


def getvalues(): # Function that read user inputs and return them to use them in initializing our model 
    ActFunction = fcmbox1.get()
    lrateValue = float(lrateEntry.get())
    EpochValue = int(EpochsEntry.get())
    biasCheck = int(biasVar.get())
    Neurons = NeuronsEntry.get()
    layers = int(LayersEntry.get())
    msg = messagebox.showinfo("Done", "Done... Now learning!")
    return ActFunction, lrateValue, EpochValue, biasCheck, Neurons, layers


# Create window
frame = Tk()
frame.geometry("500x600")
frame.title("MLP")
frame.configure(bg='#c7d1eb')



# Number of layers
Layersvar = StringVar()
layersLabel = Label(frame, textvariable=Layersvar)
Layersvar.set("Enter number of hidden layers:")
layersLabel.place(x=80, y=30)

LayersEntry = Entry(frame)
LayersEntry.place(x=270, y=30)



# Number of Neurons per layer
Neuronsvar = StringVar()
Neuronslabel = Label(frame, textvariable=Neuronsvar)
Neuronsvar.set("Enter number of neurons:")

Neuronslabel.place(x=80, y=65)
NeuronsEntry = Entry(frame)
NeuronsEntry.place(x=270, y=65)

NeuronsNoteVar = StringVar()
NeuronsNoteLabel = Label(frame, textvariable=NeuronsNoteVar)
NeuronsNoteVar.set("separate each number with comma")
NeuronsNoteLabel.place(x=160, y=98)


# Learning Rate
lratevar = StringVar()
lrateLabel = Label(frame, textvariable=lratevar)
lratevar.set('Enter learning rate:')
lrateLabel.place(x=80, y=139)

lrateEntry = Entry(frame)
lrateEntry.place(x=200, y=139)


# Epochs
Epochsvar = StringVar()
EpochsLabel = Label(frame, textvariable=Epochsvar)
Epochsvar.set('Enter Epochs:')
EpochsLabel.place(x=80, y=173)

EpochsEntry = Entry(frame)
EpochsEntry.place(x=200, y=173)

# Activation Function 
FunctionVar = StringVar()
FunctionLabel = Label(frame, textvariable=FunctionVar)
FunctionVar.set("Select Function:")
FunctionLabel.place(x=80, y=205)

fcmbox1 = ttk.Combobox(frame, value=('Sigmoid', 'Tanh'), state='readonly')
fcmbox1.place(x=200, y=205)

# Bais
biasVar = StringVar()
biasLabel = Label(frame, textvariable=biasVar)
biasVar.set("Check to add bias:")
biasLabel.place(x=136, y=248)

biasVar = IntVar()
biasCheck = Checkbutton(frame, text="Bias", variable=biasVar)
biasCheck.place(x=250, y=245)

# Train Acc
TRaccuracyVar = StringVar()
TRaccuracyLabel = Label(frame, textvariable=TRaccuracyVar)
TRaccuracyVar.set("Train Accuracy:")
TRaccuracyLabel.place(x=40, y=390)

TRaccuracyText = Text(frame, state='disabled', width=6, height=0.5)
TRaccuracyText.place(x=140, y=390)

# Test Acc 
TsaccuracyVar = StringVar()
TsaccuracyLabel = Label(frame, textvariable=TsaccuracyVar)
TsaccuracyVar.set("Test Accuracy:")
TsaccuracyLabel.place(x=280, y=390)

TsaccuracyText = Text(frame, state='disabled', width=6, height=0.5)
TsaccuracyText.place(x=370, y=390)


# Confusion Matrix
ConfVar = StringVar()
ConfLabel = Label(frame, textvariable=ConfVar)
ConfVar.set("Confusion Matrix:")
ConfLabel.place(x=10, y=515)

confusionMatrix = Text(frame, width=45, height=6)
confusionMatrix.place(x=120, y=470)
# create instance of file like object
con = PrintConsole(confusionMatrix)

# replace sys.stdout with our object
sys.stdout = con

def StartModel(): 
    TRaccuracyText.configure(state='normal')
    TRaccuracyText.delete('1.0', END)
    TRaccuracyText.configure(state='disabled')
    TsaccuracyText.configure(state='normal')
    TsaccuracyText.delete('1.0', END)
    TsaccuracyText.configure(state='disabled')
    TLabels, TData, TstLabels, TstData = fitdata()
    ActFunction, lrateValue, EpochValue, biasCheck, Neurons, layers = getvalues()
    bs = False
    if biasCheck == 1:
        bs = True
    NumNeurons = []
    Neurons = Neurons.split(',')
    for i in range(layers):
        NumNeurons.append(int(Neurons[i]))

    Setter(EpochValue, ActFunction, lrateValue, bs)

    Network = createNetwork(5, layers, NumNeurons, 3)

    TrainAcc = Fit(Network, TData, TLabels)
    TRACC = printAcc(TrainAcc,TLabels)
    TRaccuracyText.configure(state='normal')
    TRaccuracyText.delete('1.0', END)
    TRaccuracyText.insert('end', TRACC)
    TRaccuracyText.configure(state='disabled')

    TestACC = Test(Network, TstData, TstLabels)
    TSACC = printAcc(TestACC,TstLabels)
    TsaccuracyText.configure(state='normal')
    TsaccuracyText.delete('1.0', END)
    TsaccuracyText.insert('end', TSACC)
    TsaccuracyText.configure(state='disabled')
    frame.after(500,con.flush())
    ConfusionMatrix()

# CONFIRMATION BUTTON
confirmB = Button(frame, text="Run", command=StartModel, width=15, height=2, state=tk.ACTIVE, activebackground='green',font= ('Helvetica 13 bold'))
confirmB.place(x=160, y=320)

frame.mainloop()