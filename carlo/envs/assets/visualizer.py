from assets.graphics import *
from assets.entities import RectangleEntity, CircleEntity, TextEntity



class Visualizer:
    def __init__(self, width: float, height: float, ppm: int):
        # width (meters)
        # height (meters)
        # ppm is the number of pixels per meters

        self.ppm = ppm
        self.display_width, self.display_height = int(width * ppm), int(height * ppm)
        self.window_created = False
        self.visualized_imgs = []
        self.win = None

    def create_window(self, bg_color: str = "white"):
        if not self.window_created or self.win.isClosed():
            self.win = GraphWin("CARLO", self.display_width, self.display_height)
            self.win.setBackground(bg_color)
            self.window_created = True
            self.visualized_imgs = []

    def update_agents(self, agents: list, correct_pos: list=None, next_pos: list=None):
        new_visualized_imgs = []

        # Remove the movable agents from the window
        for imgItem in self.visualized_imgs:
            if imgItem["movable"]:
                imgItem["graphics"].undraw()
            else:
                new_visualized_imgs.append({"movable": False, "graphics": imgItem["graphics"]})

        # Add the updated movable agents (and the unmovable ones if they were not rendered before)
        for agent in agents:
            if isinstance(agent, TextEntity):
                img = Text(
                    Point(
                        self.ppm * agent.center.x,
                        self.display_height - self.ppm * agent.center.y,
                    ),
                    agent.text,
                )
                img.setSize(15)
                img.draw(self.win)
                # TODO(allanz): Hack: set movable=True so text is erased each iteration.
                new_visualized_imgs.append({"movable": True, "graphics": img})
            elif agent.movable or not self.visualized_imgs:
                if isinstance(agent, RectangleEntity):
                    C = [self.ppm * c for c in agent.corners]
                    img = Polygon([Point(c.x, self.display_height - c.y) for c in C])

                    # arrow
                    if agent.movable and (correct_pos and next_pos):
                        start = Point(self.ppm * agent.center.x, self.display_height - self.ppm * agent.center.y)
                        end = Point(self.ppm * correct_pos[0], self.display_height - self.ppm * correct_pos[1])
                        #print("ACTION??: ", correct_pos)
                        line = Line(start, end)
                        line.setArrow("last")
                        line.draw(self.win)
                        new_visualized_imgs.append({"movable": agent.movable, "graphics": line})
                elif isinstance(agent, CircleEntity):
                    img = Circle(
                        Point(
                            self.ppm * agent.center.x,
                            self.display_height - self.ppm * agent.center.y,
                        ),
                        self.ppm * agent.radius,
                    )
                else:
                    raise NotImplementedError
                img.setFill(agent.color)
                img.draw(self.win)
                new_visualized_imgs.append({"movable": agent.movable, "graphics": img})

        self.visualized_imgs = new_visualized_imgs

    def close(self):
        self.window_created = False
        self.win.close()
        self.visualized_imgs = []
