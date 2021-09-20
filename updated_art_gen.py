import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
from updated_utils import *
import imageio
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution() 

get_ipython().run_line_magic('matplotlib', 'inline')

model = load_vgg_model("model/vgg19.mat")

def content_cost(C,G):
    m,h,w,c = G.get_shape().as_list()

    C_unrolled = tf.transpose(a=C)
    G_unrolled = tf.reshape(G, [h*w*c])

    content = (1/(4*h*w*c))*tf.reduce_sum(input_tensor=tf.square(tf.subtract(C,G)))

    return content

tf.compat.v1.reset_default_graph()

with tf.compat.v1.Session() as sess:
    C = tf.random.normal([1,4,4,3], mean = 1, stddev = 4)
    G =  tf.random.normal([1,4,4,3], mean = 1, stddev = 4)
    content = content_cost(C,G)
    print("content = " + str(content.eval()))

def gram_matrix(M):
    Gm = tf.matmul(M, tf.transpose(a=M))
    return Gm

tf.compat.v1.reset_default_graph()

with tf.compat.v1.Session() as sess:
    M = tf.random.normal([3,2], mean=1, stddev = 4)
    Gm = gram_matrix(M)
    print("GM = " + str(Gm.eval()))

def layer_style_cost(S,G):
    m,h,w,c = G.get_shape().as_list()
    S = tf.reshape(S, shape=(h*w,c))
    G = tf.reshape(G, shape=(h*w,c))
    Gs = gram_matrix(tf.transpose(a=S))
    Gg = gram_matrix(tf.transpose(a=G))

    style_layer = (1/(4*c*c*(w*h)**2))*tf.reduce_sum(input_tensor=tf.square(tf.subtract(Gs,Gg)))
    return style_layer

tf.compat.v1.reset_default_graph()

with tf.compat.v1.Session() as sess:
    S = tf.random.normal([1,4,4,3], mean = 1, stddev = 4)
    G =  tf.random.normal([1,4,4,3], mean = 1, stddev = 4)
    layer_style = layer_style_cost(S,G)

    print("later_style = " + str(layer_style.eval()))

style_layers = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.1),
    ('conv3_1', 0.2),
    ('conv4_1', 0.4),
    ('conv5_1', 0.2),
    ]

def style_cost(model, style_layers):
    j_style=0
    for layer, coeff in style_layers:
        out = model[layer]
        S = sess.run(out)
        G = out
        style_layer = layer_style_cost(S,G)
        j_style+=coeff*style_layer
    return j_style

def total_cost(j_content, j_style, alpha=10, beta=40):
    j = alpha*j_content*beta*j_style
    return j

tf.compat.v1.reset_default_graph()

with tf.compat.v1.Session() as sess:
    j_content = np.random.randn()
    j_style = np.random.randn()
    j = total_cost(j_content, j_style)
    print("j = " + str(j))

tf.compat.v1.reset_default_graph()

sess = tf.compat.v1.InteractiveSession()

content_image = imageio.imread("images/gendeer500.jpg")
content_image = reshape_and_normalize_image(content_image)

style_image = imageio.imread("images/impressionist500x500.jpg")
style_image = reshape_and_normalize_image(style_image)

generated_image = generate_noise_image(content_image)

model = load_vgg_model("model/vgg19.mat")

sess.run(model['input'].assign(content_image))
out = model['conv4_2']
C = sess.run(out)
G = out
j_content = content_cost(C,G)

sess.run(model['input'].assign(style_image))
j_style = style_cost(model, style_layers)

j = total_cost(alpha=10, beta=40, j_content=j_content, j_style=j_style)

optimizer = tf.compat.v1.train.AdamOptimizer(2.0)
train_step = optimizer.minimize(j)


def nn(sess, input_image, iter=200):
    sess.run(tf.compat.v1.global_variables_initializer())
    generated_image = sess.run(model["input"].assign(input_image))
    for i in range(iter):
        sess.run(train_step)
        generated_image = sess.run(model["input"])

        if(i%2==0):
            q,w,e = sess.run([j, j_content, j_style])
            print("Interation " + str(i) + " :")
            print("total cost " + str(q))
            print("content cost " + str(w))
            print("style cost " + str(e))

            save_image("output/" + str(i) + ".png", generated_image)
    
    save_image("output/generated_image.jpg", generated_image)
    return  generated_image


nn(sess, generated_image)