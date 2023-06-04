import sys

import cv2
import math
import numpy as np
from PySide2 import QtWidgets, QtCore, QtGui
from PySide2.QtCore import QRect
from matplotlib import pyplot as plt

import ui_drawgraph

sys.path.append('../')
from graph_generator import TrafficDataset
import utils.traffic_utils as tutils


GRID_WIDTH = 20
LIMIT = 10
ALT = '1'
FEATS = 'color'
VISUAL_FEATS = True


class MainClass(QtWidgets.QWidget):
    def __init__(self, all_data, start_idx):
        super().__init__()
        # Plot for labels initialization
        plt.ion()
        self.fig = plt.figure('Real image | Segmented image | Overlay', frameon=False)
        self.fig.set_size_inches((640 * 3) / 100, 384 / 100)
        self.ax_labels = self.fig.add_axes([0., 0., 1., 1.])

        self.dataset = all_data
        self.ui = ui_drawgraph.Ui_SocNavWidget()
        self.ui.setupUi(self)
        self.next_index = start_idx
        self.view = None
        self.show()
        self.installEventFilter(self)
        self.load_next()


        n_features, all_features, image_features_length = tutils.get_features(ALT, FEATS)
        if VISUAL_FEATS:
            self.ui.tableWidget.setRowCount(n_features + 1)
        else:
            self.ui.tableWidget.setRowCount(len(all_features) + 1)
        self.ui.tableWidget.setColumnCount(1)
        self.ui.tableWidget.setColumnWidth(0, 200)
        self.ui.tableWidget.show()

        # Initialize table
        self.ui.tableWidget.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem('value'))
        self.ui.tableWidget.horizontalHeader().hide()
        self.ui.tableWidget.setVerticalHeaderItem(0, QtWidgets.QTableWidgetItem('type'))
        self.ui.tableWidget.setItem(0, 0, QtWidgets.QTableWidgetItem('0'))

        # Set the labels for the rows of the features table
        features_aux = self.view.graph.ndata['h'][1][:-image_features_length]
        for idx, feature in enumerate(features_aux, 1):
            self.ui.tableWidget.setVerticalHeaderItem(idx, QtWidgets.QTableWidgetItem(all_features[idx-1]))
            self.ui.tableWidget.setItem(idx, 0, QtWidgets.QTableWidgetItem('0'))

        if VISUAL_FEATS:
            for idx in range(len(all_features) + 1, n_features + 1):
                self.ui.tableWidget.setVerticalHeaderItem(idx, QtWidgets.QTableWidgetItem('VF'))
                self.ui.tableWidget.setItem(idx, 0, QtWidgets.QTableWidgetItem('0'))

    def load_next(self):
        if self.next_index >= len(self.dataset):
            print("All graphs shown")
            sys.exit(0)

        view = MyView(self.dataset[self.next_index], self.ui.tableWidget)

        if hasattr(self.view, 'view_closest_node_id'):
            view.view_closest_node_id = self.view.view_closest_node_id
            view.view_n_type = self.view.view_n_type
        if hasattr(self, 'view'):
            if self.view:
                self.view.close()
            del self.view

        self.view = view
        self.next_index += 1
        self.view.setParent(self.ui.widget)
        self.view.show()
        self.ui.widget.setFixedSize(self.view.width(), self.view.height())
        self.show()

        # Initialize table with zeros
        for idx in range(self.view.graph.ndata['h'].shape[1]):
            self.ui.tableWidget.setItem(idx, 0, QtWidgets.QTableWidgetItem('0'))
        if hasattr(self.view, 'view_closest_node_id'):
            self.view.select_node_callback()
        # Uncomment to show the label
        self._plot_label_mask(self.view.image, self.view.mask)

    def _plot_label_mask(self, label, mask):
        mask = (mask.cpu().detach().numpy() * 10).astype(np.uint8)
        label = cv2.cvtColor(cv2.resize(label, (640, 384), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

        # To cv2 format:
        mask = np.moveaxis(mask, 0, -1)
        mask = cv2.resize(mask, (640, 384), interpolation=cv2.INTER_AREA)
        mask_colour = cv2.cvtColor(cv2.applyColorMap(mask * 20, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

        mask_over = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(label, 0.4, mask_over, 0.6, 0)

        concat = np.concatenate((label, mask_colour, overlay), axis=1)

        self.ax_labels.imshow(concat, aspect='auto')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def eventFilter(self, widget, event):
        if event.type() == QtCore.QEvent.KeyPress:
            key = event.key()
            if key == QtCore.Qt.Key_Escape:
                cv2.destroyAllWindows()
                sys.exit(0)
            else:
                if key == QtCore.Qt.Key_Return:
                    self.load_next()
                elif key == QtCore.Qt.Key_Enter:
                    cv2.destroyAllWindows()
                    self.close()
                return True
        return False

    def closeEvent(self, event):
        plt.close('all')
        event.accept()  # let the window close
        sys.exit(0)


class MyView(QtWidgets.QGraphicsView):
    def __init__(self, data,  table):
        super().__init__()
        self.table = table
        self.graph = data[0]
        self.image = cv2.imread(data[3])
        self.mask = data[2]
        self.scene = QtWidgets.QGraphicsScene(self)
        self.nodeItems = dict()
        self.setFixedSize(1002, 1002)
        self.create_scene()
        self.installEventFilter(self)

    def create_scene(self):

        self.scene.setSceneRect(QtCore.QRectF(-500, -500, 1000, 1000))

        # Draw nodes and print labels
        n_features, all_features, _ = tutils.get_features(ALT, FEATS)
        all_features_iter = self.graph.ndata['h'].cpu().detach().numpy()
        for node_index, features in enumerate(all_features_iter):
            n_type = None
            if features[all_features.index('car')]:
                n_type = 'car'
                colour = QtCore.Qt.blue
                node_radius = 12
                x = (features[0] * 1000) - 500
                y = (features[1] * 1000) - 500

            elif features[all_features.index('bus')]:
                n_type = 'bus'
                colour = QtCore.Qt.green
                node_radius = 15
                x = (features[0] * 1000) - 500
                y = (features[1] * 1000) - 500

            elif features[all_features.index('truck')]:
                n_type = 'truck'
                colour = QtCore.Qt.cyan
                node_radius = 15
                x = (features[0] * 1000) - 500
                y = (features[1] * 1000) - 500

            elif features[all_features.index('person')]:
                n_type = 'person'
                colour = QtCore.Qt.black
                node_radius = 7
                x = (features[0] * 1000) - 500
                y = (features[1] * 1000) - 500

            elif features[all_features.index('grid')]:
                n_type = ''
                colour = QtCore.Qt.lightGray
                node_radius = 5
                x = (features[0] * 1000) - 500
                y = (features[1] * 1000) - 500

            else:
                colour = QtCore.Qt.white #None
                node_radius = 5 #None
                x = None
                y = None

            if x is not None:
                item = self.scene.addEllipse(x - node_radius, y - node_radius, node_radius*2,
                                             node_radius*2, brush=colour)

            else:
                print("Invalid node type")
                sys.exit(0)

            c = (x, y)
            self.nodeItems[node_index] = (item, c, n_type)

            # Print labels of the nodes
            text = self.scene.addText(n_type)
            text.setDefaultTextColor(QtCore.Qt.magenta)
            text.setPos(*c)

        # Uncomment to Draw edges
        if ALT != '0':
            edges = self.graph.edges()

            for e_id in range(len(edges[0])):
                edge_a = edges[0][e_id].item()
                edge_b = edges[1][e_id].item()

                if edge_a == edge_b:  # No self edges printed
                    continue

                ax, ay = self.nodeItems[edge_a][1]
                bx, by = self.nodeItems[edge_b][1]
                pen = QtGui.QPen()

                if edge_a < GRID_WIDTH ** 2 and edge_b < GRID_WIDTH ** 2:
                    pen.setColor(QtCore.Qt.lightGray)
                    pen.setWidth(1)
                else:
                    pen.setColor(QtCore.Qt.black)
                    pen.setWidth(2)
                self.scene.addLine(ax, ay, bx, by, pen=pen)

        self.setScene(self.scene)

    def closest_node_view(self, event_x, event_y):
        WINDOW_SIZE = 496
        closest_node = -1
        closest_node_type = -1
        x_mouse = (event_x - WINDOW_SIZE)
        y_mouse = (event_y - WINDOW_SIZE)
        old_dist = WINDOW_SIZE * 2

        for idx, node in self.nodeItems.items():
            x = node[1][0]
            y = node[1][1]
            dist = abs(x - x_mouse) + abs(y - y_mouse)
            if dist < old_dist:
                old_dist = dist
                closest_node = idx
                closest_node_type = node[2]

        return closest_node, closest_node_type

    @staticmethod
    def print_colors(feats):
        color_feats = feats[-len(tutils.COLOR_PALETTE):]
        final_color = list(tutils.COLOR_PALETTE.keys())[int(np.where(color_feats == 1.)[0])]
        print('Final_color: ', final_color)

    def select_node_callback(self):
        if not hasattr(self, 'view_closest_node_id'):
            print('We don\'t have it yet')
            return

        closest_node_id = self.view_closest_node_id
        n_type = self.view_n_type

        self.table.setItem(0, 0, QtWidgets.QTableWidgetItem(n_type))
        features = self.graph.ndata['h'][closest_node_id]
        if VISUAL_FEATS:
            print('Node features:', features)
            if FEATS == 'color' and n_type != 'grid':
                self.print_colors(features.cpu().detach().numpy())

        features_format = ['{:1.3f}'.format(x) for x in features]

        for idx, feature in enumerate(features_format):
            self.table.setItem(idx, 1, QtWidgets.QTableWidgetItem(feature))

    def eventFilter(self, widget, event):
        if event.type() == QtCore.QEvent.MouseButtonPress:
            closest_node_id, n_type = self.closest_node_view(event.x()-7, event.y()-7)
            if not n_type:
                n_type = 'grid'
            print(f'Setting node {closest_node_id}, type {n_type}')
            if n_type == -1:
                print('Not valid label')

            self.view_closest_node_id = closest_node_id
            self.view_n_type = n_type
            self.select_node_callback()

            return True
        return False


if __name__ == '__main__':
    if len(sys.argv) == 1 or len(sys.argv) > 3:
        print(f'Usage mode: python3 {sys.argv[0]} dir_to_data (start_index)')
        sys.exit(0)

    dataset = TrafficDataset("RussiaCrossroads", mode='test', project='SPADE', debug=True,
                             raw_dir=sys.argv[1], alt=ALT, alt_feats=FEATS, grid_width=GRID_WIDTH, limit=LIMIT)

    app = QtWidgets.QApplication(sys.argv)
    if len(sys.argv) > 2:
        start = int(sys.argv[2])
    else:
        start = 0

    view = MainClass(dataset, start)

    exit_code = app.exec_()
    sys.exit(exit_code)
