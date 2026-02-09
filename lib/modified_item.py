from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class ClickableImage(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        #self.setPixmap(QPixmap.fromImage(QImage("your_image.png")))
        self.pos_list = []
        

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x = event.pos().x()
            y = event.pos().y()
            x = int(x/640.0*2560.0)
            y = int(y/480.0*1440.0)
            self.pos_list.append((x,y))
            if len(self.pos_list)>2:
                self.pos_list.pop(0)

            #print(f'Clicked on pixel ({x}, {y})')
