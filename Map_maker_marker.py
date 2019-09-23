import requests
import webbrowser
from tkinter import *
from PIL import ImageTk, Image

root = Tk()
root.title(" Project ")
root.geometry("700x675")
root.resizable(width=False, height=False)


class Prjct(Frame):
    def __init__(self, master=None):

        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)
        label1 = Label(root, text="Enter the lat (-90 to 90)", font='calibri 10')
        label1.place(x=100, y=35)
        self.lat = Entry(root, width=20, bg='white')
        self.lat.place(x=450, y=35)
        label2 = Label(root, text="Enter the long (-180 to 180)", font='calibri 10')
        label2.place(x=100, y=70)
        self.longc = Entry(root, width=20, bg='white')
        self.longc.place(x=450, y=80)
        b1 = Button(root, text=("Dislay Map"), width=10, command=self.click1)
        b1.place(x=200, y=120)
        b2 = Button(root, text=("Mark"), width=10, command=self.click2)
        b2.place(x=400, y=120)
        self.canvas = Canvas(root, width=500, height=500)
        self.canvas.place(x=100, y=165)

    def click1(self):
        latco = self.lat.get()
        longco = self.longc.get()
        self.mp = latco
        self.np = longco
        url = ("https://api.mapbox.com/styles/v1/mapbox/streets-v11/static/"+longco + "," + latco + ",15,0,0/500x500?access_token=pk.eyJ1IjoibWFjaGV0ZTg5IiwiYSI6ImNqeXh3ZDd1YTEzZWwzY21sdGx2bnhwM3gifQ.MbW6tttXkNk1p24wviwHQw")
        r = requests.get(url)
        print(r)
        with open('test.png', 'wb') as f:
            f.write(r.content)
        img = ImageTk.PhotoImage(Image.open("test.png"))
        self.canvas.create_image(0, 0, anchor=NW, image=img)
        self.canvas.image = img

    def click2(self):
        X = float(self.lat.get())
        Y = float(self.longc.get())
        pc = X-float(self.mp)
        qc = Y-float(self.np)
        pp = (250+(pc*50000))
        qp = (250+(qc*50000))
        link = Label(self.canvas, text="X", font="Berlinsansfb 16 ", fg="red", cursor="hand2")
        self.canvas.create_window(pp, qp, window=link, anchor=NW)
        link.bind("<Button-1>", self.callback)

    def callback(event, self):
        webbrowser.open_new("capture.png")

app = Prjct(root)
root.mainloop()
