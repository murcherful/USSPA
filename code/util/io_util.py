import sys
import os
import numpy as np
#import cv2
import open3d as o3d

def write_image(image, path):
    with open(path, 'w') as f:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    f.write('%.8f ' % image[i, j, k])
                f.write('\n')

def write_image_as_png(image, path):  
    print('cv2 not install')
    assert(0)   
    r = image[:, :, 0:1]
    g = image[:, :, 1:2]
    b = image[:, :, 2:3]
    img = np.concatenate([b, g, r], axis=2)
    img = img * 255
    #cv2.imwrite(path, img)


def write_binary_image_as_png(image, path, scale=1):     
    print('cv2 not install')
    assert(0)
    #r = image[:, :, 0:1]
    #g = image[:, :, 1:2]
    #b = image[:, :, 2:3]
    #img = np.concatenate([b, g, r], axis=2)
    img = image * 255
    #img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    #cv2.imwrite(path, img)


def write_point_cloud(point_cloud, path):
    path += '_pc.txt'
    with open(path, 'w') as f:
        for i in range(point_cloud.shape[0]):
            for j in range(point_cloud.shape[1]):
                f.write(str(point_cloud[i, j])+';')
            f.write('\n')


def write_point_cloud_prob(point_cloud, prob, path):
    with open(path, 'w') as f:
        for i in range(point_cloud.shape[0]):
            for j in range(point_cloud.shape[1]):
                f.write(str(point_cloud[i, j])+';')
            f.write('%.3f;' % prob[i])
            f.write('\n')


def read_image(path, h, w, c):
    with open(path, 'r') as f:
        lines = f.readlines()
    res = np.zeros([h, w, c], dtype=np.float32)
    for i in range(h):
        for j in range(w):
            line = lines[i*w+j]
            line = line.split(' ')
            for k in range(c):
                res[i, j, k] = line[k]
    return res

def read_point_cloud(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    point_num = len(lines)
    res = np.zeros([point_num, 3], dtype=np.float32)
    for i in range(point_num):
        line = lines[i]
        line = line.split(';')
        for j in range(3):
            res[i, j] = float(line[j])
    return res

def write_point_cloud_as_txt_with_color_2(point_cloud, path, color1, color2, p):
    with open(path, 'w') as f:
        for i in range(point_cloud.shape[0]):
            for j in range(point_cloud.shape[1]):
                f.write(str(point_cloud[i, j])+';')
            if p[i]:
                f.write('%d;%d;%d;' % color1)
            else:
                f.write('%d;%d;%d;' % color2)
            f.write('\n')

def write_point_cloud_as_txt_with_p(point_cloud, p, match_num, path):
    p = np.expand_dims(p, axis=1)
    p = np.tile(p, [1, match_num])
    p = np.reshape(p, [-1])
    with open(path, 'w') as f:
        for i in range(point_cloud.shape[0]):
            for j in range(point_cloud.shape[1]):
                f.write(str(point_cloud[i, j])+';')
            f.write('0;' + str(int(255*p[i])) + ';0;')
            f.write('\n')

def read_point_cloud_from_txt_with_p(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    point_num = len(lines)
    res = np.zeros([point_num, 4], dtype=np.float32)
    for i in range(point_num):
        line = lines[i]
        line = line.split(';')
        for j in range(3):
            res[i, j] = float(line[j])
        res[i, 3] = float(line[4])/255.0
    return res

def write_point_cloud_with_seg(path, point_cloud, seg):
    path += '_pc.txt'
    with open(path, 'w') as f:
        for i in range(point_cloud.shape[0]):
            for j in range(point_cloud.shape[1]):
                f.write(str(point_cloud[i, j])+';')
            if seg[i] > 0.5:
                f.write('0;255;0;')
            else:
                f.write('255;0;0;')
            f.write('\n')

def write_point_cloud_with_seg_f(path, point_cloud, seg):
    path += '_pc.txt'
    with open(path, 'w') as f:
        for i in range(point_cloud.shape[0]):
            for j in range(point_cloud.shape[1]):
                f.write(str(point_cloud[i, j])+';')
            r = int(255*(1-seg[i]))
            g = int(255*seg[i])
            f.write('%d;%d;0;' % (r, g))
            f.write('\n')


def write_point_cloud_with_seg_as_ply(path, point_cloud, seg):
    path += '.ply'
    seg = seg > 0.5
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    colors = seg*np.array([[0, 1, 0]], np.float32)+(1-seg)*np.array([[1, 0, 0]], np.float32)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float))
    o3d.io.write_point_cloud(path, pcd)

def write_point_cloud_with_seg_f_as_ply(path, point_cloud, seg):
    path += '.ply'
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    colors = seg*np.array([[0, 1, 0]], np.float32)+(1-seg)*np.array([[1, 0, 0]], np.float32)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float))
    o3d.io.write_point_cloud(path, pcd)

def write_point_cloud_as_ply(path, point_cloud, with_color=False):
    path += '.ply'
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
    if with_color:
        pcd.colors = o3d.utility.Vector3dVector(point_cloud[:,3:])
    o3d.io.write_point_cloud(path, pcd)


def read_point_cloud_from_pcd(path):
    ply = o3d.io.read_point_cloud(path)
    pc = np.array(ply.points)
    return pc