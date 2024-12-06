import numpy as np
import cv2 as cv

calib_data_path = r"MultiMatrix.npz"  # Holds calibration data
calib_data = np.load(calib_data_path)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]
MARKER_SIZE = 2.3  # in cm


class trackerOverlay:
    def __init__(self,cap,marker_id,size = (2,10),position = (0,0,0),colour = (100,0,100),alpha = 1.0,line_thickness = 2,camera = cam_mat):
        self.line_thickness = line_thickness
        self.alpha = alpha
        self.camera = camera
        self.radius,self.height = size
        self.x,self.y,self.z = position
        self.pos = np.array([self.x,self.y,self.z]).reshape((3,1))
        w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        print(h,w)
        self.frameSize = (h,w)
        self.id = marker_id
        #store each corner as 3d column vector
        self.corners = np.zeros((3,4))
        self.colour = colour
        #if i'm a set of points then i will give you my dimension
        pass
    def getRotationMatrix(self,axis,angle,mode = "degrees"):
        #put angle in degrees
        #Rotation around a given axis(unit vector)
        u = (axis/np.linalg.norm(axis)).flatten()
        #turn to radians
        if mode == "radians":
            theta = angle
        elif mode == "degrees":
            theta = angle*np.pi/180
        else:
            raise(KeyError("Invalid input angle mode: '{}'.".format(mode)))
        
        ux = np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
        uxu = np.multiply.outer(u,u)
        I = np.identity(3)
        return (np.cos(theta)*I + np.sin(theta)*ux + (1-np.cos(theta))*uxu)
    
    def getRotationMatrixFromrVec(self,rvec):
        theta = np.linalg.norm(rvec)
        r = rvec/theta
        return self.getRotationMatrix(r,theta,mode = "radians")
    
    def makeWireFrame(self, base_points = 5, height_points = 1):
        #we want the base to start out at the origin, and the base should
        #be on the xz plane. Create specified number of points at radius
        #self.radius 360/base_points degrees apart and then copy this upwards
        #to form a cylinder. Then, put all the points(store as column vecs in)
        #3xn matrix through rotation matrix
        #so we rotate around the y axis to get our base
        rotationMat = self.getRotationMatrix(axis = np.array([0,1,0]), angle = 360/base_points)
        initial = np.array([self.radius,0,0]).reshape((3,1))
        #base is a circle, so the first point is connected to the last
        self.lines = []
        for i in range(1,base_points):
            initial = np.append(np.matmul(rotationMat,initial),np.array([self.radius,0,0]).reshape((3,1)),axis = 1).squeeze()
        self.lines.append((initial,'closed'))
        
        #gets the circles at different heights
        for h in range(1,height_points + 2):
            self.lines.append((initial + np.full((3,1),[[0],[h*self.height/(height_points + 1)],[0]]),'closed'))
            self.edges = 0
        
        #draw vertical lines betwixt circle points
        vert = np.concat([i[0] for i in self.lines],axis = 0)
        for colnum in range(vert.shape[1]):
            col = vert[:,colnum].reshape((1,2 + height_points,3)).squeeze().T
            self.lines.append((np.array(col),'open'))
        return

    def setFramePos(self,rvec,tvec):
        self.rvec,self.tvec = rvec,tvec
        return
    
    def recolour_draw(self,frame,colour):
        if not (hasattr(self,"rvec") or hasattr(self,"tvec")):
            return
        min_row = np.inf
        min_col = np.inf
        max_row = 0
        max_col = 0
        changed = []
        rotation = self.getRotationMatrixFromrVec(self.rvec)
        for shape in self.lines:
            pts = shape[0]
            joinstyle = True if shape[1] == 'closed' else False
            rotated = np.matmul(rotation,pts+self.pos)
            translated = rotated + self.tvec.reshape((3,1))
            captured = np.matmul(self.camera,translated)
            captured = np.rint((captured/captured[2])[:2,:]).astype('int32').T
            #print(captured.shape)
            if(self.alpha == 1):
                cv.polylines(frame,[captured],joinstyle,colour,self.line_thickness)
            else:
                #opencv does not support transparency. Adding it is costly. Don't.
                #I've added the option, but it GREATLY slows things down.
                mins = captured.min(axis = 0)
                min_row = min(max(mins[1],0),min_row,frame.shape[0])
                min_col = min(max(mins[0],0),min_col,frame.shape[1])

                maxes = captured.max(axis = 0)
                max_row = max(min(maxes[1],frame.shape[0] - 1),max_row,0)
                max_col = max(min(maxes[0],frame.shape[1] - 1),max_col,0)
                changed.append((captured,joinstyle))

        if(self.alpha != 1):
            max_row = int(max_row)
            max_col = int(max_col)
            min_row = int(min_row)
            min_col = int(min_col)
            overlay = frame[min_row:max_row + 1,min_col:max_col + 1].copy()
            for el in changed:
                capt = el[0]
                connected = el[1]
                if(overlay.size > 0):
                    cv.polylines(overlay,[capt - np.array([min_col,min_row])],connected,colour,self.line_thickness)
            frame[min_row:max_row + 1,min_col:max_col + 1] = cv.addWeighted(overlay,self.alpha,frame[min_row:max_row + 1,min_col:max_col + 1],1-self.alpha,0)

        del(self.rvec)
        del(self.tvec)

        return

    def drawInFrame(self,frame,rvec,tvec):
        rotation = self.getRotationMatrixFromrVec(rvec)
        for shape in self.lines:
            pts = shape[0]

            joinstyle = True if shape[1] == 'closed' else False
            rotated = np.matmul(rotation,pts+self.pos)
            translated = rotated + tvec.reshape((3,1))
            captured = np.matmul(self.camera,translated)
            captured = np.rint((captured/captured[2])[:2,:]).astype('int32').T
            cv.polylines(frame,[captured],joinstyle,self.colour,self.line_thickness)
        
        return
    
    def inverseTransform(self,rvec,tvec):
        #takes the transform that has been applied to our overlay, then
        #return a function that given a point, returns the inverse
        theta = -1*np.linalg.norm(rvec)
        r = rvec/theta
        reverseRot = self.getRotationMatrix(r,theta,mode = "radians")
        
        #get location relative to cylinder
        reproject_point = lambda x: np.matmul(reverseRot,x.reshape((3,-1)) - tvec.reshape((3,1))) - self.pos 
        return reproject_point
    
    def overlaps(self,marker_id,positions,margin_of_error = MARKER_SIZE/2):
        if (not (hasattr(self,"rvec") or hasattr(self,"tvec"))) or (not(marker_id in positions)):
            return False
        projectToOrigin = self.inverseTransform(self.rvec,self.tvec)
        midR = projectToOrigin(positions[marker_id]).flatten()
        midX,midY,midZ = midR
        return (np.sqrt(midX**2 + midZ**2) < self.radius+ margin_of_error) and ((midY + margin_of_error > 0) and(midY - margin_of_error < self.height))

        