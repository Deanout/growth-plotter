# Growth Plotter
Just a simple python program that reads a CSV containing dates, subscribers, and viewers and plots the graph with a linear regression.

![Demo](https://github.com/Deanout/growth-plotter/blob/main/Demo%20Plot.png)

# Setup
You'll want Python3 installed. Then in a terminal that can generate a GUI (This does not work in WSL, use PowerShell instead) you'll want to run:
```bash 
python3 -m pip install $@ scipy numpy pandas sklearn matplotlib
```

Then, to run the program, simply run this command in your terminal:
```bash
python3 ./growth.py
```
