# graphics.py
# John Zelle, 2017

import time, os, sys

try:
    import tkinter
except ImportError:
    print("Error: tkinter module not found.")
    print("You may need to install it, e.g., 'sudo apt-get install python3-tk'")
    sys.exit()

_root = tkinter.Tk()
_root.withdraw()

def update(rate=None):
    if rate:
        time.sleep(1/rate)
    _root.update()

class GraphicsError(Exception):
    pass

OBJ_ALREADY_DRAWN = "Object_Already_Drawn_Error"
UNSUPPORTED_METHOD = "Unsupported_Method_Error"
BAD_OPTION = "Bad_Option_Error"

class GraphWin(tkinter.Canvas):
    _win_instances = []

    def __init__(self, title="Graphics Window",
                 width=200, height=200, autoflush=True):

        assert type(title) == type(""), "Title must be a string"
        master = tkinter.Toplevel(_root)
        master.protocol("WM_DELETE_WINDOW", self.close)
        tkinter.Canvas.__init__(self, master, width=width, height=height,
                               highlightthickness=0, bd=0)
        self.master.title(title)
        self.pack()
        master.resizable(0,0)
        self.foreground = "black"
        self.items = []
        self.mouseX = None
        self.mouseY = None
        self.bind("<Button-1>", self._onClick)
        self.bind("<Button-2>", self._onClick)
        self.bind("<Button-3>", self._onClick)
        self.bind("<KeyPress>", self._onKey)
        self.height = int(self["height"])
        self.width = int(self["width"])
        self.autoflush = autoflush
        self._mouseCallback = None
        self.trans = None
        self.closed = False
        master.lift()
        self.lastKey = ""
        GraphWin._win_instances.append(self)
        if autoflush: _root.update()

    def __repr__(self):
        if self.isClosed():
            return "<Closed GraphWin>"
        else:
            return "GraphWin('{}', {}, {})".format(self.master.title(),
                                             self.getWidth(), self.getHeight())

    def __checkOpen(self):
        if self.closed:
            raise GraphicsError("window is closed")

    def _onKey(self, evnt):
        self.lastKey = evnt.keysym

    def _onClick(self, e):
        self.mouseX = e.x
        self.mouseY = e.y
        if self._mouseCallback:
            self._mouseCallback(Point(e.x, e.y))

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def setBackground(self, color):
        self.__checkOpen()
        self.config(bg=color)
        self.__autoflush()

    def close(self):
        if self.closed: return
        self.closed = True
        self.master.destroy()
        GraphWin._win_instances.remove(self)
        self.__autoflush()

    def isClosed(self):
        return self.closed

    def isOpen(self):
        return not self.closed

    def __autoflush(self):
        if self.autoflush:
            _root.update()

    def plot(self, x, y, color="black"):
        self.__checkOpen()
        self.create_line(x, y, x+1, y, fill=color)
        self.__autoflush()

    def flush(self):
        self.__checkOpen()
        self.update_idletasks()

    def getMouse(self):
        self.mouseX = None
        self.mouseY = None
        while self.mouseX == None or self.mouseY == None:
            self.update()
            if self.isClosed(): raise GraphicsError("getMouse in closed window")
            time.sleep(.1) # give up thread
        x,y = self.mouseX, self.mouseY
        self.mouseX = None
        self.mouseY = None
        return Point(x,y)

    def checkMouse(self):
        if self.isClosed():
            raise GraphicsError("checkMouse in closed window")
        self.update()
        if self.mouseX != None and self.mouseY != None:
            x,y = self.mouseX, self.mouseY
            self.mouseX = None
            self.mouseY = None
            return Point(x,y)
        else:
            return None

    def getKey(self):
        self.lastKey = ""
        while self.lastKey == "":
            self.update()
            if self.isClosed(): raise GraphicsError("getKey in closed window")
            time.sleep(.1) # give up thread
        key = self.lastKey
        self.lastKey = ""
        return key

    def checkKey(self):
        if self.isClosed():
            raise GraphicsError("checkKey in closed window")
        self.update()
        key = self.lastKey
        self.lastKey = ""
        return key

    def setCoords(self, x1, y1, x2, y2):
        self.trans = Transform(self.width, self.height, x1, y1, x2, y2)

    def getMouseLocation(self):
        p = self.getMouse()
        return self.trans.undraw(p.x, p.y)

    def onMouseClick(self, func):
        self._mouseCallback = func

class Transform:
    def __init__(self, w, h, x1, y1, x2, y2):
        self.win_width, self.win_height = w-1, h-1
        self.x_start, self.y_start = x1, y1
        self.x_end, self.y_end = x2, y2
        self.x_span = x2 - x1
        self.y_span = y2 - y1

    def transform(self, x, y):
        x_win = self.win_width * (x - self.x_start) / self.x_span
        y_win = self.win_height * (self.y_end - y) / self.y_span
        return x_win, y_win

    def untransform(self, x_win, y_win):
        x = self.x_start + x_win * self.x_span / self.win_width
        y = self.y_end - y_win * self.y_span / self.win_height
        return x, y

class GraphicsObject:
    def __init__(self, options):
        self.canvas = None
        self.id = None
        config = {}
        for option in options:
            if option in self.config:
                config[option] = options[option]
        self.config = config

    def _draw(self, canvas, options):
        self.canvas = canvas
        if canvas.isClosed(): raise GraphicsError("draw in closed window")
        self.id = self._draw_kernel(canvas, options)
        if canvas.autoflush:
            _root.update()
        return self

    def _draw_kernel(self, canvas, options):
        pass

    def undraw(self):
        if not self.canvas: return
        if not self.canvas.isClosed():
            self.canvas.delete(self.id)
            if self.canvas.autoflush:
                _root.update()
        self.canvas = None
        self.id = None

    def move(self, dx, dy):
        self._move(dx, dy)
        if self.canvas and not self.canvas.isClosed():
            trans = self.canvas.trans
            if trans:
                x1, y1 = trans.transform(0,0)
                x2, y2 = trans.transform(dx,dy)
                dx, dy = x2-x1, y2-y1
            self.canvas.move(self.id, dx, dy)
            if self.canvas.autoflush:
                _root.update()

    def _move(self, dx, dy):
        pass

    def setFill(self, color):
        self._reconfig("fill", color)

    def setOutline(self, color):
        self._reconfig("outline", color)

    def setWidth(self, width):
        self._reconfig("width", width)

    def _reconfig(self, option, val):
        if self.canvas and not self.canvas.isClosed():
            self.canvas.itemconfig(self.id, {option:val})
            if self.canvas.autoflush:
                _root.update()

class Point(GraphicsObject):
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __repr__(self):
        return "Point({}, {})".format(self.x, self.y)

    def clone(self):
        return Point(self.x, self.y)

    def getX(self): return self.x
    def getY(self): return self.y

    def _draw_kernel(self, canvas, options):
        x,y = self.x, self.y
        if canvas.trans:
            x,y = canvas.trans.transform(x,y)
        return canvas.create_line(x,y,x+1,y, options)

    def _move(self, dx, dy):
        self.x += dx
        self.y += dy

class Line(GraphicsObject):
    def __init__(self, p1, p2):
        self.p1 = p1.clone()
        self.p2 = p2.clone()

    def __repr__(self):
        return "Line({}, {})".format(str(self.p1), str(self.p2))

    def clone(self):
        return Line(self.p1, self.p2)

    def getP1(self): return self.p1.clone()
    def getP2(self): return self.p2.clone()
    def getCenter(self):
        return Point((self.p1.x+self.p2.x)/2.0, (self.p1.y+self.p2.y)/2.0)

    def _draw_kernel(self, canvas, options):
        p1, p2 = self.p1, self.p2
        if canvas.trans:
            p1 = canvas.trans.transform(p1.x, p1.y)
            p2 = canvas.trans.transform(p2.x, p2.y)
        return canvas.create_line(p1,p2, options)

    def _move(self, dx, dy):
        self.p1.move(dx,dy)
        self.p2.move(dx,dy)

class Rectangle(GraphicsObject):
    def __init__(self, p1, p2):
        self.p1 = p1.clone()
        self.p2 = p2.clone()

    def __repr__(self):
        return "Rectangle({}, {})".format(str(self.p1), str(self.p2))

    def clone(self):
        return Rectangle(self.p1, self.p2)

    def getP1(self): return self.p1.clone()
    def getP2(self): return self.p2.clone()
    def getCenter(self):
        return Point((self.p1.x+self.p2.x)/2.0, (self.p1.y+self.p2.y)/2.0)

    def _draw_kernel(self, canvas, options):
        p1, p2 = self.p1, self.p2
        if canvas.trans:
            p1 = canvas.trans.transform(p1.x, p1.y)
            p2 = canvas.trans.transform(p2.x, p2.y)
        return canvas.create_rectangle(p1,p2, options)

    def _move(self, dx, dy):
        self.p1.move(dx,dy)
        self.p2.move(dx,dy)

class Oval(Rectangle):
    def __repr__(self):
        return "Oval({}, {})".format(str(self.p1), str(self.p2))
    def clone(self):
        return Oval(self.p1, self.p2)
    def _draw_kernel(self, canvas, options):
        p1, p2 = self.p1, self.p2
        if canvas.trans:
            p1 = canvas.trans.transform(p1.x, p1.y)
            p2 = canvas.trans.transform(p2.x, p2.y)
        return canvas.create_oval(p1,p2, options)

class Circle(Point):
    def __init__(self, center, radius):
        p1 = Point(center.x-radius, center.y-radius)
        p2 = Point(center.x+radius, center.y+radius)
        self.center = center.clone()
        self.radius = radius
        self.rect = Rectangle(p1,p2)

    def __repr__(self):
        return "Circle({}, {})".format(str(self.center), str(self.radius))

    def clone(self):
        return Circle(self.center, self.radius)

    def getRadius(self):
        return self.radius

    def getCenter(self):
        return self.center.clone()

    def _draw_kernel(self, canvas, options):
        return self.rect._draw_kernel(canvas, options)

    def _move(self, dx, dy):
        self.center.move(dx,dy)
        self.rect.move(dx,dy)

class Polygon(GraphicsObject):
    def __init__(self, *points):
        self.points = [p.clone() for p in points]

    def __repr__(self):
        return "Polygon" + str(tuple(p for p in self.points))

    def clone(self):
        return Polygon(*self.points)

    def getPoints(self):
        return list(map(Point.clone, self.points))

    def _draw_kernel(self, canvas, options):
        points = self.points
        if canvas.trans:
            points = [canvas.trans.transform(p.x, p.y) for p in points]
        return canvas.create_polygon(points, options)

    def _move(self, dx, dy):
        for p in self.points:
            p.move(dx,dy)

class Text(Point):
    def __init__(self, p, text):
        Point.__init__(self, p.x, p.y)
        self.text = text
        self.setJustification("center")
        self.setFont("helvetica", 12, "normal")

    def __repr__(self):
        return "Text({}, '{}')".format(str(self.p1), self.getText())

    def clone(self):
        other = Text(self, self.text)
        other.config = self.config.copy()
        return other

    def getText(self):
        return self.text

    def setText(self, text):
        self.text = text
        self._reconfig("text", text)

    def setFace(self, face):
        self._reconfig_font("family", face)

    def setSize(self, size):
        self._reconfig_font("size", size)

    def setStyle(self, style):
        self._reconfig_font("weight", style)

    def setJustification(self, value):
        self._reconfig("justify", value)

    def _reconfig_font(self, option, value):
        font = list(eval(self.canvas.itemcget(self.id, "font")))
        # Simplified font logic for robustness
        try:
            if option == "family": font[0] = value
            elif option == "size": font[1] = value
            elif option == "weight": font[2] = value
        except IndexError:
            pass # Fails gracefully if font format is unexpected
        self._reconfig("font", tuple(font))

    def _draw_kernel(self, canvas, options):
        p = self.p
        if canvas.trans:
            p = canvas.trans.transform(p.x,p.y)
        return canvas.create_text(p, options)

    def _move(self, dx, dy):
        self.p.move(dx, dy)

class Entry(Rectangle):
    def __init__(self, p, width):
        p2 = p.clone(); p2.move(width*10, 20)
        Rectangle.__init__(self, p, p2)
        self.entry = None

    def _draw_kernel(self, canvas, options):
        p1 = self.p1
        if canvas.trans:
            p1 = canvas.trans.transform(p1.x, p1.y)
        self.entry = tkinter.Entry(canvas.master)
        self.entry.place(x=p1.x, y=p1.y)

    def getText(self):
        return self.entry.get()

class Image(GraphicsObject):
    def __init__(self, p, pixmap):
        self.p = p.clone()
        self.pixmap = pixmap
        self.img = None

    def _draw_kernel(self, canvas, options):
        p = self.p
        if canvas.trans:
            p = canvas.trans.transform(p.x, p.y)
        self.img = tkinter.PhotoImage(master=canvas, data=self.pixmap)
        return canvas.create_image(p.x, p.y, image=self.img)

    def getAnchor(self):
        return self.p.clone()

    def _move(self, dx, dy):
        self.p.move(dx, dy)

if __name__ == "__main__":
    win = GraphWin("My Test Window", 400, 400)
    win.setBackground("lightgray")
    l = Line(Point(100,100), Point(200,200))
    l.draw(win)
    c = Circle(Point(300, 150), 50)
    c.setFill("red")
    c.draw(win)
    win.getMouse()
    win.close()