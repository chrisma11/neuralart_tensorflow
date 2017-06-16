import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
import os
import time

CONTENT_IMG = './images/harvard.jpg'
STYLE_IMG = './images/starry_night.jpg'
OUTOUT_DIR = './results'
OUTPUT_IMG = 'results.png'
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
# Noise ratio of the initial input image
NOISE_RATIO = 0.6
# The emphasis put on style loss in relation to content loss in total loss computation
STYLE_STRENGTH = 500
ITERATION = 1000

# Fully-connected layers at the end are left out because we will not use them
VGG19_LAYERS = [
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
]
# Layers that abstract content information
CONTENT_LAYERS =['conv4_2']
# Use the relation between these layers to represent style
STYLE_LAYERS=['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1']
# Mean pixel value VGG used to train
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

def build_vgg19(data_path, shape):
    """
    Extract relevent layers and weights from the raw VGG19 model
    """
    data = scipy.io.loadmat(data_path)
    weights = data['layers'][0]
    current = tf.Variable(np.zeros(shape).astype('float32'))
    net = {'input': current}
    for i in range(len(VGG19_LAYERS)):
        layer = VGG19_LAYERS[i]
        if layer.startswith('conv'):
            W, b = weights[i][0][0][0][0]
            W = tf.constant(W)
            b = b.reshape(-1)
            conv = tf.nn.conv2d(current, W, strides=(1, 1, 1, 1), padding='SAME')
            # combine conv layer and relu layer for simplicity
            current = tf.nn.relu(tf.nn.bias_add(conv, b))
        elif layer.startswith('pool'):
            current = tf.nn.avg_pool(current, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        net[layer] = current
    return net

def blend_noise(content_img):
    """
    Returns a noise image intermixed with the content image
    """
    noise_img = np.random.uniform(-20, 20, content_img.shape).astype('float32')
    output_image = NOISE_RATIO * noise_img + (1 - NOISE_RATIO) * content_img
    return output_image

def content_loss(sess, content_img, model, layer):
    sess.run([model['input'].assign(content_img)])
    ref = model[layer]
    val = sess.run(ref)
    filter_s = val.shape[1] * val.shape[2]
    filter_n = val.shape[3]
    # loss is defined as 0.5 * tf.nn.l2_loss(ref - val) in the paper
    loss = (1./(2 * filter_n * filter_s)) * tf.nn.l2_loss(ref - val)
    return loss

def gram_matrix(ref, val, filter_s, filter_n):
    """
    The feature correlations are given by the Gram matrix, which is the
    inner product between the vectorised feature map i and j in layer l
    """
    matrix = tf.reshape(ref, (filter_s,filter_n))
    gram = tf.matmul(tf.transpose(matrix), matrix)
    matrix_val = val.reshape(filter_s,filter_n)
    gram_val = np.dot(matrix_val.T, matrix_val)
    return gram, gram_val

def style_loss(sess, style_img, model, layer):
    sess.run([model['input'].assign(style_img)])
    ref = model[layer]
    val = sess.run(ref)
    filter_s = val.shape[1] * val.shape[2]
    filter_n = val.shape[3]
    G, A = gram_matrix(ref, val, filter_s, filter_n)
    loss = (1./(4 * filter_s**2 * filter_n**2)) * tf.nn.l2_loss(G - A)
    return loss

def read_images(content, style):
    """
    Returns processed content image and style image
    """
    content_img = scipy.misc.imread(content).astype(np.float)
    target_shape = content_img.shape
    # resize style image to fit the content image
    style_img = scipy.misc.imread(style).astype(np.float)
    style_img = scipy.misc.imresize(style_img, (target_shape[0], target_shape[1]))
    # add an extra dimension
    content_img = np.reshape(content_img, ((1,) + target_shape)) - MEAN_VALUES
    style_img = np.reshape(style_img, ((1,) + target_shape)) - MEAN_VALUES
    
    return content_img, style_img

def save_image(path, image):
    image = image + MEAN_VALUES
    image = np.clip(image[0], 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)

def main():
    
    if not os.path.exists(OUTOUT_DIR):
        os.mkdir(OUTOUT_DIR)
    
    content_img, style_img = read_images(CONTENT_IMG, STYLE_IMG)
    
    net = build_vgg19(VGG_MODEL, content_img.shape)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # sum of content/style losses of every layers
    cost_content = sum(map(lambda l: content_loss(sess, content_img, net, l), CONTENT_LAYERS))
    cost_style = sum(map(lambda l: style_loss(sess, style_img, net, l), STYLE_LAYERS))

    cost_total = cost_content + STYLE_STRENGTH * cost_style
  
    train_step = tf.train.AdamOptimizer(2.).minimize(cost_total)

    sess.run(tf.global_variables_initializer())
    sess.run(net['input'].assign(blend_noise(content_img)))

    for i in range(ITERATION):
        sess.run(train_step)
        print 'Iteration %d' % i
        if i % 200 ==0:
            result_img = sess.run(net['input'])
            print 'cost:', sess.run(cost_total)
            save_image(os.path.join(OUTOUT_DIR,'%d.png'%i),result_img)
    
    sess.close()
    
    output_path = os.path.join(OUTOUT_DIR,OUTPUT_IMG)
    save_image(output_path, result_img)


if __name__ == '__main__':
    start_time = time.clock()
    main()
    print("-- %s seconds --" % (time.clock() - start_time))
