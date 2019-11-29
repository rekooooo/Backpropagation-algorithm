import  tkinter
from  tkinter import *
from  tkinter.ttk import Combobox
from tkinter.ttk import Checkbutton
import numpy as np
import Training
import matplotlib.pyplot as plt



#TODO : make Window there Size 800*200
window  = Tk()
# that is for window
window.title("Computational Intelligence Tasks")
window.geometry("1800x1000")

##############################################
#TODO : make entry for number of hidden layers

label = Label( text="No of hidden layers:")
label.place(x=100,y=100)

label.config(font=("Century Gothic", 10))
n_h_entry = Entry()

n_h_entry.place(x=400, y=100)

#################################################
#TODO : make entry for number of neurons in each hidden layer

label = Label( text="No of neurons in each hidden layers:")
label.place(x=100,y=150)

label.config(font=("Century Gothic", 10))
nn_entry = Entry()

nn_entry.place(x=400, y=150)

#################################################
#TODO : make entry for learning rate

label = Label( text="learning rate:")
label.place(x=100,y=200)

label.config(font=("Century Gothic", 10))
L_entry = Entry()

L_entry.place(x=400, y=200)

#################################################
#TODO : make entry for epochs

label = Label( text="epochs:")
label.place(x=100,y=250)

label.config(font=("Century Gothic", 10))
epochs_entry = Entry()

epochs_entry.place(x=400, y=250)
###############################################
#TODO : make entry To get Bais

var = IntVar()
bias_check =Checkbutton(window,text="Bais",variable=var)

bias_check.place(x=400, y=300)
#####################################################
#TODO : make entry for activation function

label = Label( text="activation function:")
label.place(x=100,y=350)

label.config(font=("Century Gothic", 10))
AF = ["sigmoid", "tanh"]
activationFunctionCombo = Combobox(window,values=AF,state='readonly')
activationFunctionCombo.current(0)
activationFunctionCombo.place(x=400,y=350)
######################################################
#TODO : Run button
def callFunc():
    results, accuracy = Training.intialize(int(n_h_entry.get()), float(L_entry.get()), int(epochs_entry.get()), var.get(),
                                         activationFunctionCombo.get(), nn_entry.get())


    label = Label(text="Confusion Matrix: \n"+str(results), bg="pink")
    label.place(x=300, y=450)
    label.config(font=("Courier", 15))

    label2 = Label(text="accuracy is: "+str(accuracy), bg="pink")
    label2.place(x=300, y=550)

    label2.config(font=("Courier", 15))

    fig = plt.figure()

    ax = fig.add_subplot(111)
    cax = ax.matshow(results)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + ['class1', 'class2', 'class3'])
    ax.set_yticklabels([''] + ['class1', 'class2', 'class3'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


    return

buttonRun = Button(text="Run..." ,bg="pink", command=callFunc, width=20, height=2)


buttonRun.place(x=300, y=400)

window.configure(background='light gray')
window.mainloop()