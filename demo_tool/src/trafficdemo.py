import sys
import math

from PyQt6.QtWidgets import QApplication, QWidget, QPushButton
from PyQt6.QtCore import Qt, QPoint, QRect
from PyQt6.QtGui import QPixmap, QPen, QPainter, QColor, QBrush, QImage
from PyQt6 import uic
from PyQt6.QtWidgets import QMessageBox

from shapely.geometry import Polygon

sys.path.append("../")
from trafficAPI import *


class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("ui/gui.ui", self)

        self.colour_buttons = None
        self.alert = None
        self.background_img = None
        self.input_pixmap = None
        self.output_pixmap = None
        self.t_API = None
        self.init_ui()

        width = IMAGE_SIZE[0] * 2 + 100
        height = self.frameGeometry().height()
        self.setFixedSize(width, height)
        self.redraw()

    def init_ui(self):
        # Create colours layout
        row = 0
        column = 0
        self.colour_buttons = []
        for colour in COLOR_PALETTE:
            button = QPushButton(text='', parent=self, objectName=colour)
            button.setCheckable(True)
            button.setStyleSheet(f'background-color: {colour}')
            button.clicked.connect(self.onColourButtonClicked)
            if colour == "Black":
                button.setChecked(True)
            self.colour_buttons.append(button)
            self.gridLayoutColours.addWidget(button, row, column)
            column += 1
            if column == 2:
                row += 1
                column = 0

        self.colour = 'Black'
        self.day_time = [0, 0, 0]

        # Init tool buttons
        self.addCarButton.clicked.connect(self.onAddCarButtonClicked)
        self.addPersonButton.clicked.connect(self.onAddPersonButtonClicked)
        self.addVanButton.clicked.connect(self.onAddVanButtonClicked)
        self.addBusButton.clicked.connect(self.onAddBusButtonClicked)
        self.deleteButton.clicked.connect(self.onDeleteButtonClicked)
        self.deleteAllButton.clicked.connect(self.onDeleteAllButtonClicked)
        self.addTimeButton.clicked.connect(self.onAddTimeButtonClicked)
        self.items = []
        self.tool = None
        self.tool_buttons = [self.addCarButton, self.addPersonButton, self.addVanButton, self.addBusButton,
                             self.deleteButton]

        # Init alert message
        self.alert = QMessageBox()
        self.alert.setWindowTitle("Hello!")
        self.alert.setText("Select a tool please!")

        # Init backgrounds of the two pixmap
        self.background_img = QPixmap("assets/background2.png")
        self.background_img = self.background_img.scaled(IMAGE_SIZE[0], IMAGE_SIZE[1])
        self.input_pixmap = self.background_img.copy()
        self.output_pixmap = QPixmap(IMAGE_SIZE[0], IMAGE_SIZE[1])
        (self.init_coords, self.end_coords) = (QPoint(), QPoint())

        # Init traffic API
        self.t_API = TrafficAPI()

    # ###############################################################
    # Mouse events
    #################################################################
    def mousePressEvent(self, event):
        if self.tool is None:
            self.alert.exec()
        else:
            self.init_coords = self.widget2input(event.pos().x(), event.pos().y())

    def mouseMoveEvent(self, event):
        self.end_coords = self.widget2input(event.pos().x(), event.pos().y())
        if self.end_coords.isNull() or self.init_coords.isNull():
            return

        self.updateAndRedraw()

        painter = QPainter(self.input_pixmap)
        painter.setBrush(Qt.GlobalColor.black)
        painter.setOpacity(0.3)
        rect = QRect(self.init_coords, self.end_coords)
        painter.drawRect(rect.normalized())
        self.redraw()

    def mouseReleaseEvent(self, event):
        # Get end_coords
        self.end_coords = self.widget2input(event.pos().x(), event.pos().y())
        if self.end_coords.isNull() or self.init_coords.isNull():
            return
        # Transform init and end coords into x, y, width and height
        x = min(self.init_coords.x(), self.end_coords.x()) / IMAGE_SIZE[0]
        y = min(self.init_coords.y(), self.end_coords.y()) / IMAGE_SIZE[1]
        width = int(math.fabs(self.init_coords.x() - self.end_coords.x())) / IMAGE_SIZE[0]
        height = int(math.fabs(self.init_coords.y() - self.end_coords.y())) / IMAGE_SIZE[1]
        print(x, y, width, height)

        visual_features = np.zeros(len(COLOR_PALETTE))
        visual_features[list(COLOR_PALETTE.keys()).index(self.colour)] = 1.

        if self.tool is None:
            pass
        elif self.tool == 'add_car':
            self.items.append(
                TrafficItem(conf=0.9, xywh=[x, y, width, height], cls=SPADE_LABELS['car'],
                            visualFeatures=visual_features, dayTime=self.day_time))
        elif self.tool == 'add_person':
            self.items.append(
                TrafficItem(conf=0.9, xywh=[x, y, width, height], cls=SPADE_LABELS['person'],
                            visualFeatures=visual_features, dayTime=self.day_time))
        elif self.tool == 'add_van':
            self.items.append(
                TrafficItem(conf=0.9, xywh=[x, y, width, height], cls=SPADE_LABELS['truck'],
                            visualFeatures=visual_features, dayTime=self.day_time))
        elif self.tool == 'add_bus':
            self.items.append(
                TrafficItem(conf=0.9, xywh=[x, y, width, height], cls=SPADE_LABELS['bus'],
                            visualFeatures=visual_features, dayTime=self.day_time))
        elif self.tool == 'delete':
            self.delete_items(x, y, width, height)
        else:
            print(f'Unhandled tool {self.tool}')
            sys.exit(-1)
        self.updateAndRedraw()
        self.generate_output()

    # ###############################################################
    # Buttons tools events
    #################################################################
    def onAddCarButtonClicked(self):
        for tool in self.tool_buttons:
            if tool != self.addCarButton:
                tool.setChecked(False)
        if self.addCarButton.isChecked():
            self.tool = 'add_car'
        else:
            self.tool = None
        print(self.tool)

    def onAddPersonButtonClicked(self):
        for tool in self.tool_buttons:
            if tool != self.addPersonButton:
                tool.setChecked(False)
        if self.addPersonButton.isChecked():
            self.tool = 'add_person'
        else:
            self.tool = None
        print(self.tool)

    def onAddVanButtonClicked(self):
        for tool in self.tool_buttons:
            if tool != self.addVanButton:
                tool.setChecked(False)
        if self.addVanButton.isChecked():
            self.tool = 'add_van'
        else:
            self.tool = None
        print(self.tool)

    def onAddBusButtonClicked(self):
        for tool in self.tool_buttons:
            if tool != self.addBusButton:
                tool.setChecked(False)
        if self.addBusButton.isChecked():
            self.tool = 'add_bus'
        else:
            self.tool = None
        print(self.tool)

    def onDeleteButtonClicked(self):
        for tool in self.tool_buttons:
            if tool != self.deleteButton:
                tool.setChecked(False)
        if self.deleteButton.isChecked():
            self.tool = 'delete'
        else:
            self.tool = None
        print(self.tool)

    def onDeleteAllButtonClicked(self):
        self.items = []
        self.updateAndRedraw()

    def onAddTimeButtonClicked(self):
        self.day_time = [int(self.addDayTime.time().hour()), int(self.addDayTime.time().minute()), 0]
        print(self.day_time)
        self.generate_output()

    # ###############################################################
    # Buttons colours events
    #################################################################
    def onColourButtonClicked(self):
        sender = self.sender()
        self.colour = sender.objectName()
        for c_button in self.colour_buttons:
            if c_button.objectName() != self.colour:
                c_button.setChecked(False)
        print(self.colour)
    # ###############################################################
    # Utility methods
    #################################################################

    def updateAndRedraw(self):
        self.input_pixmap = self.background_img.copy()
        painter = QPainter(self.input_pixmap)
        painter.setOpacity(0.3)
        for it in self.items:
            x = int(it.xywh[0] * IMAGE_SIZE[0])
            y = int(it.xywh[1] * IMAGE_SIZE[1])
            w = int(it.xywh[2] * IMAGE_SIZE[0])
            h = int(it.xywh[3] * IMAGE_SIZE[1])
            begin = QPoint(x, y)
            end = QPoint(x + w, y + h)

            cls_name = YOLO_NAMES[it.cls]
            if cls_name == "car":
                painter.setBrush(Qt.GlobalColor.red)
            elif cls_name == "truck":
                painter.setBrush(Qt.GlobalColor.blue)
            elif cls_name == "person":
                painter.setBrush(Qt.GlobalColor.black)
            elif cls_name == "bus":
                painter.setBrush(Qt.GlobalColor.green)
            else:
                print('Not valid item to draw')
                sys.exit(0)

            rect = QRect(begin, end)
            painter.drawRect(rect.normalized())

        self.redraw()

    def delete_items(self, x, y, width, height):
        polygon = Polygon([(x, y), (x + width, y), (x + width, y + height), (x, y + height)])
        new_items = []
        for o in self.items:
            o_x, o_y, o_width, o_height = tuple(o.xywh)
            other = Polygon([(o_x, o_y), (o_x + o_width, o_y), (o_x + o_width, o_y + o_height), (o_x, o_y + o_height)])
            intersection = polygon.intersection(other)
            if intersection.area <= 0:
                new_items.append(o)
        self.items = new_items

    def redraw(self):
        self.inputLabel.setPixmap(self.input_pixmap)
        self.outputLabel.setPixmap(self.output_pixmap)

    def widget2input(self, x, y):
        x_ret = x - self.inputLabel.x()
        y_ret = y - self.inputLabel.y()
        if x_ret < 0 or x_ret >= self.input_pixmap.width():
            return QPoint()
        elif y_ret < 0 or y_ret >= self.input_pixmap.height():
            return QPoint()
        return QPoint(x_ret, y_ret)

    def generate_output(self):
        # Add this first element to the list since it is always detected (false positive as reference)
        data = [[TrafficItem(conf=0.1,
                             xywh=[0., 0., 0., 0.],
                             cls=SPADE_LABELS['car'],
                             visualFeatures=[0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                             dayTime=self.day_time)]]
        data[0] = data[0] + self.items
        output = self.t_API.generate_one_output(data)

        if output is not None:
            output = ((output + 1) * 255/2).astype(np.uint8)
            height, width, channel = output.shape
            bytesPerLine = 3 * width
            image = QImage(output.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
            self.output_pixmap = QPixmap(image)
            self.redraw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
