from threading import Thread
from queue import Queue
import cv2
import sys

class videoFrameHelper:
    def __init__(self ,VCObject):
        self.video = VCObject
        self.frameQue = Queue(maxsize=256)
        self.threadTerminated = False

        frameHandlingThread = Thread(target=self.handleFrames)
        frameHandlingThread.start()

    def handleFrames(self):
        while True:

            #check if thread has been terminated
            if self.threadTerminated:
                return
            
            #check if que is full
            if not self.frameQue.full():
                (present, frame) = self.video.read()

                #if frame isnt avaliable then the video is over
                if not present:
                    self.terminateThread()
                    return
                
                #append frame
                self.frameQue.put(frame)

    def frameQueEmpty(self):
        return (self.frameQue.qsize() <= 0)

    def terminateThread(self):
        self.threadTerminated = True
    
    def getNextFrame(self):
        return self.frameQue.get()

