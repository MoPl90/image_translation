from tensorflow.keras import backend as K

import numpy as np
import scipy.ndimage as ndi
import random


""" Range recommendation
  RotationAngle_Xaxis: +/- 2
  Shift_Xaxis = +/- 0.02
  Zoom   0.95 - 1.05
  StretchFactor  0.95 - 1.05
  ShearFactor  +/- 0.03 """

class augmentor():

    def __init__(self, rotationAngle_Xaxis = 0, rotationAngle_Yaxis = 0, rotationAngle_Zaxis = 0,
                 shift_Xaxis = 0, shift_Yaxis = 0, shift_Zaxis = 0,
                 stretch_Xaxis = 0, stretch_Yaxis = 0, stretch_Zaxis = 0,
                 shear_NormalXAxis = 0, shear_NormalYAxis = 0, shear_NormalZAxis = 0,
                 zoom = 1, flip = 0):

        self.rotateAngleX = rotationAngle_Xaxis
        self.rotateAngleY = rotationAngle_Yaxis
        self.rotateAngleZ = rotationAngle_Zaxis
        self.shiftValueX = shift_Xaxis
        self.shiftValueY = shift_Yaxis
        self.shiftValueZ = shift_Zaxis
        self.stretchValueX = stretch_Xaxis
        self.stretchValueY = stretch_Yaxis
        self.stretchValueZ = stretch_Zaxis
        self.shearValueX = shear_NormalXAxis
        self.shearValueY = shear_NormalYAxis
        self.shearValueZ = shear_NormalZAxis
        self.zoomValue = zoom
        self.flipBool = flip

        #additional settings for scipy.ndimage.interpolation.affine_transform
        self.fillMode = 'constant'
        self.cValue = 0.0
        self.interpolationOrderX = 0  
        self.interpolationOrderY = 3  

        #stores number of transformations per step
        self.numbTransforms = []


    def meanNumberTransforms(self):
        if 0 < len(self.numbTransforms):
            avgTransforms = sum(self.numbTransforms) / len(self.numbTransforms)
        else:
            return 0
        self.numbTransforms = []
        return avgTransforms


    def augmentXY(self, X, Y, maxTransforms=2):

        # Define how many transforms
        # numTransforms = random.randint(0, maxTransforms)
        numTransforms = maxTransforms

        seeds = []
        for x in range(0, 14):
            seeds.append(random.randint(0, 100)*random.randint(0, 1)/100)
            seeds[x] *= [-1, 1][random.randrange(2)]

        # Limit the number of transformations per input
        keep = [0] * len(seeds)
        for j in range(0, numTransforms):
            keep[random.randint(0, len(seeds)-1)] = 1

        # Set all seeds to 0 which shouldn't be applied
        seeds = np.multiply(seeds,keep)
        self.numbTransforms.append(sum(seeds != 0.0))

        # Augment the input matrix X (label map)
        X = self.augment(X, seeds, self.interpolationOrderX)

        # Augment the output matrix Y 
        Y = self.augment(Y, seeds, self.interpolationOrderY)

        return X, Y


    def augment(self, matrix, seeds, order):

        # Rotation
        if ((seeds[0] != 0) or (seeds[1] != 0) or (seeds[2] != 0)):
            angle_x = self.rotateAngleX * seeds[0]
            angle_y = self.rotateAngleY * seeds[1]
            angle_z = self.rotateAngleZ * seeds[2]
            matrix = self.rotation_3D(matrix, angle_x, angle_y, angle_z, order=order)

        # Stretch
        if ((seeds[3] != 0) or (seeds[4] != 0) or (seeds[5] != 0)):
            stretch_x = 1 + self.stretchValueX * seeds[3]
            stretch_y = 1 + self.stretchValueY * seeds[4]
            stretch_z = 1 + self.stretchValueZ * seeds[5]
            matrix = self.stretch_3D(matrix, stretch_x, stretch_y, stretch_z, order=order)

        # Translate
        if ((seeds[6] != 0) or (seeds[7] != 0) or (seeds[8] != 0)):
            shift_Xaxis = self.shiftValueX * seeds[3]
            shift_Yaxis = self.shiftValueY * seeds[4]
            shift_Zaxis = self.shiftValueZ * seeds[5]
            
            matrix = self.shift_3D(matrix, [shift_Xaxis, shift_Yaxis, shift_Zaxis], shift_axis='x', order=order)
                
        # Shear
        if ((seeds[9] != 0) or (seeds[10] != 0) or (seeds[11] != 0)):
            shear_x = self.shearValueX * seeds[9]
            shear_y = self.shearValueY * seeds[10]
            shear_z = self.shearValueZ * seeds[11]
            matrix = self.shear_3D(matrix, [shear_x, shear_y, shear_z], order=order)

        # Flip
        if (self.flipBool and (seeds[12] > 0.5)):
            matrix = self.flip_3D(matrix)

        # Zoom 
        if (seeds[13] != 0):
            zoom = 1 + [-1, 1][np.random.randint(0,1)] * self.zoomValue * seeds[13]
            matrix = self.zoom_3D(matrix, zoom, order=order)
 
        return matrix


    def transform_matrix_offset_center_3D(self, matrix, x, y, z):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        o_z = float(z) / 2 + 0.5

        offset_matrix = np.array([[1, 0, 0, o_x], [0, 1, 0, o_y], [0, 0, 1, o_z], [0, 0, 0, 1]])
        reset_matrix = np.array([[1, 0, 0, -o_x], [0, 1, 0, -o_y], [0, 0, 1, -o_z], [0, 0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)

        return transform_matrix


    def apply_transform_3D(self, x, transform_matrix, channel_index=0, order=0):

        x = np.rollaxis(x, channel_index, 0)

        final_affine_matrix = transform_matrix[:3, :3]
        final_offset = transform_matrix[:3, 3]

        channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix, final_offset, order=order, mode=self.fillMode, cval=self.cValue) for x_channel in x]

        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_index+1)

        return x


    def rotation_3D(self, inputMatrix, ang_x, ang_y, ang_z, row_index=0, col_index=1, depth_index=2, channel_index=3, order=0):

        theta_x = np.pi / 180 * ang_x
        theta_y = np.pi / 180 * ang_y
        theta_z = np.pi / 180 * ang_z

        rot_matrix_x = np.array([[1, 0, 0, 0],
                                 [0, np.cos(theta_x), -np.sin(theta_x), 0],
                                 [0, np.sin(theta_x), np.cos(theta_x), 0],
                                 [0, 0, 0, 1]])

        rot_matrix_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y), 0],
                                 [0, 1, 0, 0],
                                 [-np.sin(theta_y), 0, np.cos(theta_y), 0],
                                 [0, 0, 0, 1]])

        rot_matrix_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0, 0],
                                 [np.sin(theta_z), np.cos(theta_z), 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])

        #perform matrix multiplication with rotation matrices in all directions for final transformation (for angle=0 -> identity matrix)
        rotation_matrix = np.matmul(rot_matrix_x, rot_matrix_y)
        rotation_matrix = np.matmul(rotation_matrix, rot_matrix_z)

        h, w, d = inputMatrix.shape[row_index], inputMatrix.shape[col_index], inputMatrix.shape[depth_index]
        transform_matrix = self.transform_matrix_offset_center_3D(rotation_matrix, h, w, d)
        rotatedMatrix = self.apply_transform_3D(inputMatrix, transform_matrix, channel_index, order)

        return rotatedMatrix


    def stretch_3D(self, inputMatrix, stretchFactor_x, stretchFactor_y, stretchFactor_z, row_index=0, col_index=1, depth_index=2, channel_index=3, order=0):

        #stretch matrix, stretch can be performed in all dimensions at once
        stretch_matrix = np.array([[stretchFactor_x, 0, 0, 0],
                                   [0, stretchFactor_y, 0, 0],
                                   [0, 0, stretchFactor_z, 0],
                                   [0, 0, 0, 1]])

        h, w, d = inputMatrix.shape[row_index], inputMatrix.shape[col_index], inputMatrix.shape[depth_index]
        transform_matrix = self.transform_matrix_offset_center_3D(stretch_matrix, h, w, d)
        stretchedMatrix = self.apply_transform_3D(inputMatrix, transform_matrix, channel_index, order)

        return stretchedMatrix

    def shift_3D(self, inputMatrix, shift, row_index=0, col_index=1, depth_index=2, channel_index=3, shift_axis='z',
        fill_mode='nearest', cval=0., order=0):

        # shift is multiplication factor of the size of the image
        t = inputMatrix.shape[depth_index] * shift

        translation_matrix = np.array([ [1, 0, 0, t[0]],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])


        translation_matrix = np.matmul(translation_matrix,
                                       np.array([[1, 0, 0, 0],
                                                 [0, 1, 0, t[1]],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]]))


        translation_matrix = np.matmul(translation_matrix,
                                       np.array([[1, 0, 0, 0],
                                                 [0, 1, 0, 0],
                                                 [0, 0, 1, t[2]],
                                                 [0, 0, 0, 1]]))

        # transform_matrix = translation_matrix  # no need to do offset
        # rotatedMatrix = apply_transform(inputMatrix, transform_matrix, channel_index, fill_mode, cval)
        shiftedMatrix = self.apply_transform_3D(inputMatrix, translation_matrix, channel_index, order)
        return shiftedMatrix

    def shear_3D(self, inputMatrix, shearFactors, row_index=0, col_index=1, depth_index=2, channel_index=3,
                 fill_mode='nearest', cval=0., order=0):

        shear_matrix = np.array([[1, shearFactors[0], 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])


        shear_matrix = np.matmul(shear_matrix, 
                                 np.array([[1, 0, 0, 0],
                                           [0, 1, shearFactors[1], 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]]))


        shear_matrix = np.matmul(shear_matrix, 
                                 np.array([[1, 0, shearFactors[2], 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]]))

        h, w, d = inputMatrix.shape[row_index], inputMatrix.shape[col_index], inputMatrix.shape[depth_index]
        transform_matrix = self.transform_matrix_offset_center_3D(shear_matrix, h, w, d)
        shearedMatrix = self.apply_transform_3D(inputMatrix, transform_matrix, channel_index, order)

        return shearedMatrix

    def flip_3D(self, inputMatrix, flipAxis = 0):
        return np.flip(inputMatrix, flipAxis)


    def zoom_3D(self, inputMatrix, zoom, row_index=0, col_index=1, depth_index=2, channel_index=3,
                fill_mode='nearest', cval=0., order=0):

        # zoom factor will be multiplied to the current volume


        zoom_matrix = np.array([[zoom, 0, 0, 0],
                                [0, zoom, 0, 0],
                                [0, 0, zoom, 0],
                                [0, 0, 0, 1]])

        h, w, d = inputMatrix.shape[row_index], inputMatrix.shape[col_index], inputMatrix.shape[depth_index]
        transform_matrix = self.transform_matrix_offset_center_3D(zoom_matrix, h, w, d)
        zoomedMatrix = self.apply_transform_3D(inputMatrix, transform_matrix, channel_index, order)

        return zoomedMatrix