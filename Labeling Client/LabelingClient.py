import sys
import threading

from NatNetClient import NatNetClient
import numpy as np
import math

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import socket
import struct

import queue as queue

import time


# performs all processing and runs the neural network
# the labeled data is streamed to outputIP at outputPort
class ThreadedWorker(object):
    resolutionPcm = 4
    resolutionPmm = resolutionPcm / 10
    # image size in cm
    imageSize = 25
    # max y in mm
    ultimateY = 120
    # number of pixels
    nop = imageSize * resolutionPcm
    # center point in image
    zzz = [nop / 2, 0, nop - (8 * resolutionPcm)]

    q = queue.LifoQueue(maxsize=0)
    logged = False

    outputPort = 1512
    outputSocket = None

    tfDevice = '/cpu:0'
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    inputTensor = None
    outputTensor = None
    sess = None

    markerLabels = None
    rShapesAreStreamed = True
    lastRData = np.array([[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]], dtype='f')
    lastTimeStep = None
    maxTimeBetweenRFrames = 0.5 #0.004 * 3
    maxLegalDistForRs = 0.006
    # delta for R_Shape detection
    delta = 0.00475
    # R_Shape distances
    d12 = 0.04067110384179894
    d13 = 0.03997714900977185
    d14 = 0.014055661378941353
    d23 = 0.02587136293308418
    d24 = 0.047480735670875227
    d34 = 0.03835638333555752


    # load the graph of the neural network from a .pb file
    def load_graph(self, frozen_graph_filename):
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="prefix",
                op_dict=None,
                producer_op_list=None
            )
            return graph

    def __init__(self, interval=1):
        self.interval = interval

        print("loading tensorflow", tf.__version__)

        self.outputSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print("loading neural network")
        with tf.device(self.tfDevice):
            graph = self.load_graph("./labelNetwork.pb")
            # We access the input and output nodes
            self.inputTensor = graph.get_tensor_by_name('prefix/conv2d_1_input:0')
            self.outputTensor = graph.get_tensor_by_name('prefix/output_node0:0')

            self.sess = tf.Session(graph=graph, config=self.config)

        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True  # Daemonize thread
        thread.start()  # Start the execution

    def normalizeY(self, coordinates):
        yt = coordinates[1::3]
        yt = (yt / self.ultimateY)
        coordinates[1::3] = yt
        return coordinates

    # filter data which is outside the depth image bounds
    def filterData(self, coordinates, transformedAndShiftedData):  # coordinates in mm
        # filter coordinates and apply the same filter to transformedAndShiftedData
        xt = coordinates[0::3]
        yt = coordinates[1::3]
        zt = coordinates[2::3]
        xtas = transformedAndShiftedData[0::3]
        ytas = transformedAndShiftedData[1::3]
        ztas = transformedAndShiftedData[2::3]

        xtAfterFilter = []
        ytAfterFilter = []
        ztAfterFilter = []
        xTransformedAndShiftedDataOutput = []
        yTransformedAndShiftedDataOutput = []
        zTransformedAndShiftedDataOutput = []
        for i in range(len(xt)):
            # if the data is outside the grid size it does not belong to the hand
            if xt[i] < 0 or xt[i] >= self.nop or zt[i] < 0 or zt[i] >= self.nop or yt[i] < -self.ultimateY or yt[
                i] > self.ultimateY:
                continue

            xtAfterFilter.append(xt[i])
            ytAfterFilter.append(yt[i])
            ztAfterFilter.append(zt[i])
            xTransformedAndShiftedDataOutput.append(xtas[i])
            yTransformedAndShiftedDataOutput.append(ytas[i])
            zTransformedAndShiftedDataOutput.append(ztas[i])

        cNew = np.array([xtAfterFilter, ytAfterFilter, ztAfterFilter])
        cNew = cNew.T.reshape(-1)
        tasNew = np.array(
            [xTransformedAndShiftedDataOutput, yTransformedAndShiftedDataOutput, zTransformedAndShiftedDataOutput])
        tasNew = tasNew.T.reshape(-1)

        return {"coordinates": cNew, "transformedAndShiftedData": tasNew}

    # create depth image of size nop x nop
    def createImage(self, coordinates):  # coordinates in mm
        image = np.zeros((1, self.nop, self.nop, 1))

        coordinatesF = coordinates  # still floats
        coordinates = coordinates.astype(np.int)

        for j in range(0, coordinates.shape[0]):
            # x z plane image with y value
            image[0][coordinates[j][0]][coordinates[j][2]][0] = [coordinatesF[j][1]][0]  # y values are normalized

        return image

    def unit_vector(self, vector):
        return vector / np.linalg.norm(vector)

    # perform transformation to hand coordinates by calculating new basis and performing change of basis
    def transformToHandCoordinates(self, rs, data):
        r1 = rs[0]
        r2 = rs[1]
        r3 = rs[2]
        r4 = rs[3]

        m12 = np.array([(r1[0] + r2[0]) / 2, (r1[1] + r2[1]) / 2, (r1[2] + r2[2]) / 2])
        m14 = np.array([(r1[0] + r4[0]) / 2, (r1[1] + r4[1]) / 2, (r1[2] + r4[2]) / 2])
        m23 = np.array([(r3[0] + r2[0]) / 2, (r3[1] + r2[1]) / 2, (r3[2] + r2[2]) / 2])
        m34 = np.array([(r3[0] + r4[0]) / 2, (r3[1] + r4[1]) / 2, (r3[2] + r4[2]) / 2])

        # find three linear independent vectors vx, vy, vz
        vx = self.unit_vector(m23 - m14)
        vy = self.unit_vector(np.cross(vx, (m12 - m34)))
        vz = self.unit_vector(np.cross(vx, vy))

        baseOld = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
        baseNew = np.array([vx, vy, vz]).T
        cob = np.linalg.solve(baseNew, baseOld)

        rotated = np.dot(cob, data.T)
        return {"rotated": rotated.T, "rotationMatrix": cob}

    def getDistanceNP(self, marker, data):
        return np.sqrt(np.sum(np.square(np.subtract(data, marker)), axis=1))

    def getDistanceNP2(self, data0, data1):
        return np.sqrt(np.sum(np.square(np.subtract(data0, data1))))

    def run(self):
        while True:
        ##########################################
        ####### Receive Data From Listener #######
        ##########################################
            # get first element of the queue and clear queue to minimize delay
            data = self.q.get()
            with self.q.mutex:
                self.q.queue.clear()

            if not self.logged:
                print("Receiving Frame Data, Running Neural Network and Streaming Labeled Data to", outputIP)
                self.logged = True

            frame = {}
            for i in range(len(self.markerLabels)):
                frame[self.markerLabels[i]] = data[i]

            for i in range(len(self.markerLabels), len(data)):
                frame["Unlabeled_" + str(i - len(self.markerLabels))] = data[i]


        ################################################
        ####### Rigid Body Detection/ Extraction #######
        ################################################
            rs = {}
            timestep = time.time()
            # can the last R_Shaoes be tracked or do we have to rerun the deteciton
            doFindNewRS = True
            if ((not self.lastTimeStep == None) and timestep <= self.lastTimeStep + self.maxTimeBetweenRFrames and timestep >= self.lastTimeStep):
                dataNP = np.array(data)
                doFindNewRS = False
                # try to track the last R_Shapes by finding the nearest neighbors
                rmseList = [float("inf")] * self.lastRData.shape[0]
                indexList = [-1] * self.lastRData.shape[0]
                queue = list(range(self.lastRData.shape[0]))
                while len(queue) > 0:
                    i = queue.pop(0)
                    candidate = self.lastRData[i]
                    dist = np.array(
                        [dataNP[:, 0] - candidate[0], dataNP[:, 1] - candidate[1], dataNP[:, 2] - candidate[2]])
                    dist = dist.T
                    dist = np.sqrt(np.mean(np.square(dist), axis=1))
                    minI = np.argmin(dist, axis=0)

                    foundNN = False
                    while not foundNN:
                        # if all distances ar inf, all data is labeled to closer neighbors
                        if dist[i] == float("inf"):
                            break
                        # if there is no label found yet for the nearest neighbor we found one
                        if indexList[i] == -1:
                            indexList[i] = minI
                            rmseList[i] = dist[minI]
                            foundNN = True
                        # if the new candidate is closer than the previous nearest neighbor, set it as nn and run the other one again
                        elif rmseList[i] > dist[minI]:
                            queue.append(indexList[i])
                            indexList[i] = minI
                            rmseList[i] = dist[minI]
                            foundNN = True
                        # if there is already another marker closer to the R, set its distance to inf and find the 2nd nearest neighbor
                        else:
                            dist[minI] = float("inf")
                            minI = np.argmin(dist, axis=0)

                # Check for max distance, if they are too distant we have to rerun the R_Shape detection
                if not (all(i <= self.maxLegalDistForRs for i in rmseList)):
                    doFindNewRS = True

                # save last R data for the next run
                self.lastRData = dataNP[indexList]
            if doFindNewRS:
                rCandidates = []
                for indexR1, r1c in enumerate(data):
                    distances = self.getDistanceNP(r1c, data)

                    f1 = distances[distances >= self.d14 - self.delta]
                    if len(f1[f1 <= self.d24 + self.delta]) < 3:
                        continue

                    # find candidates for r4
                    for indexR4, r4c in enumerate(data):
                        distanceR14 = distances[indexR4]
                        if (distanceR14 >= self.d14 - self.delta) and (distanceR14 <= self.d14 + self.delta):
                            # find candidates for r2
                            for indexR2, r2c in enumerate(data):
                                if not ((distances[indexR2] >= self.d12 - self.delta) and (
                                        distances[indexR2] <= self.d12 + self.delta)):
                                    continue
                                distanceR24 = self.getDistanceNP2(r4c, r2c)
                                if (distanceR24 >= self.d24 - self.delta) and (distanceR24 <= self.d24 + self.delta):
                                    # find candidates for r3
                                    for indexR3, r3c in enumerate(data):
                                        if not ((distances[indexR3] >= self.d13 - self.delta) and (
                                                distances[indexR3] <= self.d13 + self.delta)):
                                            continue
                                        distanceR23 = self.getDistanceNP2(r2c, r3c)
                                        if (distanceR23 >= self.d23 - self.delta) and (distanceR23 <= self.d23 + self.delta):

                                            # verify by checking the other distances
                                            distanceR34 = self.getDistanceNP2(r3c, r4c)
                                            if (distanceR34 >= self.d34 - self.delta) and (distanceR34 <= self.d34 + self.delta):
                                                rCandidates.append(data[indexR1])
                                                rCandidates.append(data[indexR2])
                                                rCandidates.append(data[indexR3])
                                                rCandidates.append(data[indexR4])
                self.lastRData = np.array(rCandidates)
            self.lastTimeStep = time.time()
            for i in range(int(self.lastRData.shape[0] / 4)):
                rs["r" + str(len(rs))] = [self.lastRData[i * 4 + 0], self.lastRData[i * 4 + 1], self.lastRData[i * 4 + 2], self.lastRData[i * 4 + 3]]

            frameList = list(frame.values())


        ##################################
        ####### Filter by Distance #######
        ##################################
            images = np.array([])
            transformedAndShiftedDataMemory = []
            rotationResultMemory = []
            rcenterMemory = []
            for i in range(len(rs)):
                rshape = rs["r" + str(i)]
                # find the center of the R_Shape
                rcenter = [(rshape[0][0] + rshape[1][0] + rshape[2][0] + rshape[3][0]) / 4,
                           (rshape[0][1] + rshape[1][1] + rshape[2][1] + rshape[3][1]) / 4,
                           (rshape[0][2] + rshape[1][2] + rshape[2][2] + rshape[3][2]) / 4]

                # the data for the neural network's grid, which does not contain the R_Shape data and data that is too far away from the R_Shape
                dataForGrid = []
                # remove rs itself
                for j in range(len(frameList)):
                    if ((frameList[j][0] == rshape[0][0] and frameList[j][1] == rshape[0][1] and frameList[j][2] ==
                         rshape[0][2]) or
                            (frameList[j][0] == rshape[1][0] and frameList[j][1] == rshape[1][1] and frameList[j][
                                2] == rshape[1][2]) or
                            (frameList[j][0] == rshape[2][0] and frameList[j][1] == rshape[2][1] and frameList[j][
                                2] == rshape[2][2]) or
                            (frameList[j][0] == rshape[3][0] and frameList[j][1] == rshape[3][1] and frameList[j][
                                2] == rshape[3][2])):
                        continue

                    # filter by distance to remove markers that are too far away from R_Shape
                    # 0.212 m distance is to far away
                    if math.sqrt((rcenter[0] - frameList[j][0]) ** 2 +
                                 (rcenter[1] - frameList[j][1]) ** 2 +
                                 (rcenter[2] - frameList[j][2]) ** 2) >= 0.212:
                        continue


        ###################################
        ####### Shift to new Center #######
        ###################################
                    # shift data to new basis center
                    shifted = [frameList[j][0] - rcenter[0], frameList[j][1] - rcenter[1],
                               frameList[j][2] - rcenter[2]]
                    dataForGrid.append(shifted)

                # shift R to new basis center
                rzero = [[rshape[0][0] - rcenter[0], rshape[0][1] - rcenter[1], rshape[0][2] - rcenter[2]],
                         [rshape[1][0] - rcenter[0], rshape[1][1] - rcenter[1], rshape[1][2] - rcenter[2]],
                         [rshape[2][0] - rcenter[0], rshape[2][1] - rcenter[1], rshape[2][2] - rcenter[2]],
                         [rshape[3][0] - rcenter[0], rshape[3][1] - rcenter[1], rshape[3][2] - rcenter[2]]]


        #############################################
        ####### Transform to hand coordinates #######
        #############################################
                # transform to hand coordinates
                npData = np.array(dataForGrid)
                if npData.shape[0] == 0:
                    continue
                rotationResult = self.transformToHandCoordinates(rzero, npData)
                dataForGrid = rotationResult["rotated"]

                # remember the data in meter and independent from resolution for nearest neighbour search after running the neural network
                transformedAndShiftedData = dataForGrid.reshape(-1)

                # from meter to millimeter
                dataForGrid = dataForGrid * 1000


        #########################################################
        ####### Create Depth Image for the Neural Network #######
        #########################################################
                # from millimeter to image resolution and centered to the image center 'zzz'
                dataForGrid = (dataForGrid * self.resolutionPmm) + self.zzz

                # reshape for further processing
                dataForGrid = dataForGrid.reshape(-1)

                # remove resolutionPmm from y again, it will be normalized to [-1,1]
                yt = dataForGrid[1::3]
                yt = (yt / self.resolutionPmm)
                dataForGrid[1::3] = yt

                # filter by distance to remove markers that are too far away from R_Shape
                filterResult = self.filterData(dataForGrid, transformedAndShiftedData)
                dataForGrid = filterResult["coordinates"]
                transformedAndShiftedData = filterResult["transformedAndShiftedData"]

                # normalize y
                dataForGrid = self.normalizeY(dataForGrid)

                # reshape again
                dataForGrid = dataForGrid.reshape(-1, 3)
                transformedAndShiftedData = transformedAndShiftedData.reshape(-1, 3)

                # create image for NN
                image = self.createImage(dataForGrid)

                if images.shape[0] == 0:
                    images = image

                else:
                    images = np.append(images, image, axis=0)
                transformedAndShiftedDataMemory.append(transformedAndShiftedData)
                rotationResultMemory.append(rotationResult)
                rcenterMemory.append(rcenter)


            if images.shape[0] == 0:
                continue


        #####################################
        ####### Run the Neural Netwok #######
        #####################################
            pred_y_all = self.sess.run(self.outputTensor, feed_dict={self.inputTensor: images})

            for hand in range(pred_y_all.shape[0]):
                pred_y = pred_y_all[hand]
                transformedAndShiftedData = transformedAndShiftedDataMemory[hand]
                rotationResult = rotationResultMemory[hand]
                rcenter = rcenterMemory[hand]


        #########################################################
        ####### Find the Nearest Neighbors for the Labels #######
        #########################################################
                # get labels from neural network by nearest neighbour
                prediction = np.array(pred_y)
                # change to meter again
                prediction = prediction / 1000
                prediction = prediction.reshape(21, 2)

                # save the indices of the nearest neighbors and their distances
                rmseList = [float("inf")] * 21
                indexList = [-1] * 21
                # find the nearest candidate for each label prediction
                queue = list(range(transformedAndShiftedData.shape[0]))
                while len(queue) > 0:
                    i = queue.pop(0)
                    candidate = transformedAndShiftedData[i]

                    dist = np.array([prediction[:, 0] - candidate[0], prediction[:, 1] - candidate[2]])
                    dist = dist.T
                    dist = np.sqrt(np.mean(np.square(dist), axis=1))
                    minI = np.argmin(dist, axis=0)

                    foundNN = False
                    while not foundNN:
                        # if all distances ar inf, all data is labeled to closer neighbors
                        if dist[minI] == float("inf"):
                            break
                        # if there is no label found yet for the nearest neighbor we found one
                        if indexList[minI] == -1:
                            indexList[minI] = i
                            rmseList[minI] = dist[minI]
                            foundNN = True
                        # if the new candidate is closer than the previous nearest neighbor, set it as nn and run the other one again
                        elif rmseList[minI] > dist[minI]:
                            queue.append(indexList[minI])
                            indexList[minI] = i
                            rmseList[minI] = dist[minI]
                            foundNN = True
                        # if there is already another marker closer to the nearest label, set its distance to inf and find the 2nd nearest neighbor
                        else:
                            dist[minI] = float("inf")
                            minI = np.argmin(dist, axis=0)

                # set the coordinates of the labeled markers, if there is not a candidate for each label, set its coordinates to (0, 0, 0)
                labeledList = np.zeros((21, 3))
                indexList = np.array(indexList)
                indexListSetOn = np.where(indexList >= 0)
                labeledList[indexListSetOn] = transformedAndShiftedData[indexList[indexListSetOn]]


        ####################################################################
        ####### Classify whether the data is left or right hand data #######
        ####################################################################
                # is left or right hand?
                # the thumb mcp of right hand data has negative y value
                isRightHand = float(0)
                if labeledList[3, 1] < 0:
                    isRightHand = float(1)


        #############################################################
        ####### Transform and shift back to world coordinates #######
        #############################################################
                labeledList = np.dot(np.linalg.inv(rotationResult["rotationMatrix"]),
                                     labeledList.T).T
                labeledList = labeledList + rcenter
                labeledList = list(labeledList.reshape(-1))


        ####################################################
        ####### Stream the Labeled Data to Output IP #######
        ####################################################
                # add whether data is right handed or not
                labeledList.insert(0, isRightHand)
                # stream to unity
                buf = struct.pack('%sf' % len(labeledList), *labeledList)
                self.outputSocket.sendto(buf, (outputIP, self.outputPort))



# receives the marker definition and saves the marker labels
def markerDefinitionListener(markerSetName, names):
    if markerSetName == "all":
        print("marker definition received")
        removePrefix = (lambda x: str(x).split(":")[1] if len(str(x).split(":")) > 1 else str(x))
        # remove "all:" from marker labels
        for i in range(len(names)):
            names[i] = removePrefix(names[i])

        worker.markerLabels = names


# receives the marker data, and pass to worker thread
def markerDataListener(data):
    worker.q.put(data)


# This will create a new NatNet client
streamingClient = NatNetClient()

streamingClient.markerDefinitionListener = markerDefinitionListener
streamingClient.markerDataListener = markerDataListener

worker = ThreadedWorker()
print("==========================================================================")
outputIP = "192.168.56.1"
if len(sys.argv) > 1:
    streamingClient.serverIPAddress = sys.argv[1]
if len(sys.argv) > 2:
    outputIP = sys.argv[2]
if len(sys.argv) > 3:
    if sys.argv[3] == "gpu":
        worker.tfDevice = '/gpu:0'
        worker.config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, device_count={'GPU': 4})
        print("Using Device " + worker.tfDevice)
else:
    print("No Motive IP or Output IP was given. Run with parameters to hand over IPs! Run for Example:")
    print("sudo python3 LabelingClient.py <motiveIP> <outputIP>")
    print("sudo python3 LabelingClient.py 192.168.56.1 192.168.56.1")
    print("==========================================================================")
print("Motive IP", streamingClient.serverIPAddress)
print("Output IP", outputIP)
print("==========================================================================")



print("starting natnet streaming client")
streamingClient.run()

