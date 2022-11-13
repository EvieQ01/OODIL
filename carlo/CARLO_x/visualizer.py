import pdb
from graphics import *
from graphics import _root
from entities import RectangleEntity, CircleEntity, RingEntity
from PIL import Image, ImageTk, ImageGrab  # For Windows & OSx
# import pyscreenshot as ImageGrab # For Linux
class Visualizer:
    def __init__(self, width: float, height: float, ppm: int):
        # width (meters)
        # height (meters)
        # ppm is the number of pixels per meters
        
        self.ppm = ppm
        self.display_width, self.display_height = int(width*ppm), int(height*ppm)
        self.window_created = False
        self.visualized_imgs = []
                
    def gettimg(self):
        # x=_root.winfo_rootx()+self.win.winfo_x()
        # print(x)
        # y=_root.winfo_rooty()+self.win.winfo_y()
        # print(y)
        # x1=x+self.win.winfo_width()
        # print(x1)
        # y1=y+self.win.winfo_height()
        # print(y1)
        # ImageGrab.grab(bbox=(x,y,x1,y1)).save("em.png")
        return ImageGrab.grab(bbox=(70,200,780,800))#.save("em.png") # witdh 360 height 300

    def create_window(self, bg_color: str = 'gray80'):
        if not self.window_created or self.win.isClosed():
            self.win = GraphWin('CARLO', self.display_width, self.display_height)
            self.win.setBackground(bg_color)
            self.window_created = True
            self.visualized_imgs = []
            
    def update_agents(self, agents: list):
        new_visualized_imgs = []
        
        # Remove the movable agents from the window
        for imgItem in self.visualized_imgs:
            if imgItem['movable']:
                imgItem['graphics'].undraw()
            else:
                new_visualized_imgs.append({'movable': False, 'graphics': imgItem['graphics']})
                
        # Add the updated movable agents (and the unmovable ones if they were not rendered before)
        for agent in agents:
            if agent.movable or not self.visualized_imgs:
                if isinstance(agent, RectangleEntity):
                    C = [self.ppm*c for c in agent.corners]
                    img = Polygon([Point(c.x, self.display_height-c.y) for c in C])
                elif isinstance(agent, CircleEntity):
                    img = Circle(Point(self.ppm*agent.center.x, self.display_height - self.ppm*agent.center.y), self.ppm*agent.radius)
                elif isinstance(agent, RingEntity):
                    img = CircleRing(Point(self.ppm*agent.center.x, self.display_height - self.ppm*agent.center.y), self.ppm*agent.inner_radius, self.ppm*agent.outer_radius)
                else:
                    raise NotImplementedError
                img.setFill(agent.color)
                img.draw(self.win)
                new_visualized_imgs.append({'movable': agent.movable, 'graphics': img})
                
        self.visualized_imgs = new_visualized_imgs

    def close(self):
        self.window_created = False
        self.win.close()
        self.visualized_imgs = []